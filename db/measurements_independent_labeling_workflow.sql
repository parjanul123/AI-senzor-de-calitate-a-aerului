-- Workflow SQL pentru etichetare independenta si activare train Random Forest/SVM fara data leakage.
-- Ruleaza in Supabase SQL Editor.
-- Presupune ca tabela public.measurements are cheia primara coloana id.

begin;

-- 1) Asigura coloanele necesare in measurements
alter table public.measurements
    add column if not exists quality_label text,
    add column if not exists quality_label_source text;

alter table public.measurements
    add constraint if not exists measurements_quality_label_check
    check (quality_label is null or quality_label in ('good', 'moderate', 'poor'));

alter table public.measurements
    add constraint if not exists measurements_quality_label_source_check
    check (
        quality_label_source is null
        or quality_label_source in (
            'manual',
            'expert_review',
            'external_aqi_standard',
            'lab_reference',
            'independent_sensor_fusion'
        )
    );

-- 2) Creeaza tabela de review pentru etichetare independenta
create table if not exists public.measurements_quality_review (
    measurement_id text primary key,
    quality_label text not null check (quality_label in ('good', 'moderate', 'poor')),
    quality_label_source text not null check (
        quality_label_source in (
            'manual',
            'expert_review',
            'external_aqi_standard',
            'lab_reference',
            'independent_sensor_fusion'
        )
    ),
    reviewer text,
    review_notes text,
    reviewed_at timestamptz not null default now()
);

commit;

-- 3) Selecteaza randuri pentru etichetare (exemplu: ultimele 200 ne-etichetate)
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
limit 200;

-- 4) Exemple de etichetare manuala (inlocuieste ID-urile)
insert into public.measurements_quality_review
    (measurement_id, quality_label, quality_label_source, reviewer, review_notes)
values
    ('REPLACE_ID_1', 'good', 'manual', 'operator_1', 'Conditii bune observate in teren'),
    ('REPLACE_ID_2', 'moderate', 'expert_review', 'specialist_aer', 'Valori moderate validate expert'),
    ('REPLACE_ID_3', 'poor', 'manual', 'operator_1', 'Eveniment poluare confirmat')
on conflict (measurement_id) do update
set
    quality_label = excluded.quality_label,
    quality_label_source = excluded.quality_label_source,
    reviewer = excluded.reviewer,
    review_notes = excluded.review_notes,
    reviewed_at = now();

-- 5) Sincronizeaza etichetele aprobate in measurements
update public.measurements m
set
    quality_label = r.quality_label,
    quality_label_source = r.quality_label_source
from public.measurements_quality_review r
where m.id::text = r.measurement_id;

-- 6) Verifica daca datasetul este pregatit pentru train supervised
-- Conditii minime in cod: valid_rows >= 10 si class_count >= 2
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

-- 7) Verifica distributia claselor
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
