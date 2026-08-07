-- Add independent supervised-label support for leakage-free RF/SVM training.
-- Run in Supabase SQL Editor.

begin;

alter table public.measurements
    add column if not exists quality_label text,
    add column if not exists quality_label_source text;

-- Optional quality checks
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

-- Temporary bootstrap for already-labeled rows.
-- IMPORTANT: keep only if these labels were assigned independently by humans/external standards.
update public.measurements
set quality_label_source = 'manual'
where quality_label is not null
  and quality_label_source is null;

commit;

-- Readiness check used by the training pipeline (minimum 10 rows, at least 2 classes)
select
    count(*) as valid_rows,
    count(distinct quality_label) as class_count,
    min(created_at) as first_labeled_at,
    max(created_at) as last_labeled_at
from public.measurements
where quality_label in ('good', 'moderate', 'poor')
  and quality_label_source in (
      'manual',
      'expert_review',
      'external_aqi_standard',
      'lab_reference',
      'independent_sensor_fusion'
  );
