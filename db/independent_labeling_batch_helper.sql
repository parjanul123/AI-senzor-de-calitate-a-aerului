-- Helper rapid pentru etichetare independenta in batch.
-- Scop: sa pregatesti strict-mode training (/train cu allow_derived_label_fallback=false).
-- Ruleaza in Supabase SQL Editor.

-- 1) Preconditii (coloane + tabela de review)
-- Daca nu ai rulat deja migrarile, ruleaza mai intai:
--   db/measurements_supervised_labels_migration.sql
--   db/measurements_independent_labeling_workflow.sql

-- 2) Selecteaza un lot de randuri ne-etichetate pentru review manual
-- Ajusteaza LIMIT dupa nevoie (minim 10 randuri etichetate, minim 2 clase diferite).
select
    m.id,
    m.created_at,
    m.temperature,
    m.humidity,
    m.pm25,
    m.pm10,
    m.co2,
    m.voc
from public.measurements m
left join public.measurements_quality_review r on r.measurement_id = m.id::text
where r.measurement_id is null
order by m.created_at desc
limit 30;

-- 3) Completeaza manual etichetele pentru randurile selectate
-- IMPORTANT: etichetele trebuie sa fie independente de feature-uri (review uman/expert/standard extern).
-- Inlocuieste REPLACE_* cu valori reale.
insert into public.measurements_quality_review
    (measurement_id, quality_label, quality_label_source, reviewer, review_notes)
values
    ('REPLACE_ID_1', 'good', 'manual', 'reviewer_1', 'eticheta independenta'),
    ('REPLACE_ID_2', 'moderate', 'manual', 'reviewer_1', 'eticheta independenta'),
    ('REPLACE_ID_3', 'poor', 'manual', 'reviewer_1', 'eticheta independenta')
on conflict (measurement_id) do update
set
    quality_label = excluded.quality_label,
    quality_label_source = excluded.quality_label_source,
    reviewer = excluded.reviewer,
    review_notes = excluded.review_notes,
    reviewed_at = now();

-- 4) Sincronizeaza etichetele aprobate in tabela measurements
update public.measurements m
set
    quality_label = r.quality_label,
    quality_label_source = r.quality_label_source
from public.measurements_quality_review r
where m.id::text = r.measurement_id;

-- 5) Verifica readiness pentru train supervised strict-mode
select
    count(*) as valid_rows,
    count(distinct m.quality_label) as class_count,
    min(m.created_at) as first_labeled_at,
    max(m.created_at) as last_labeled_at
from public.measurements m
where m.quality_label in ('good', 'moderate', 'poor')
  and m.quality_label_source in (
      'manual',
      'expert_review',
      'external_aqi_standard',
      'lab_reference',
      'independent_sensor_fusion'
  );

-- 6) Verifica distributia claselor (ideal: toate cele 3 clase prezente)
select
    m.quality_label,
    count(*) as rows_count
from public.measurements m
where m.quality_label in ('good', 'moderate', 'poor')
  and m.quality_label_source in (
      'manual',
      'expert_review',
      'external_aqi_standard',
      'lab_reference',
      'independent_sensor_fusion'
  )
group by m.quality_label
order by m.quality_label;
