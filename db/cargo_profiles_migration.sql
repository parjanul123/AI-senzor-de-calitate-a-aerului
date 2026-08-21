-- Extinde tabela public.profiles pentru profiluri de transport frigorific.
-- Ruleaza in Supabase SQL Editor inainte de folosirea endpointurilor /transport/profiles.

alter table public.profiles
    add column if not exists profile_id text,
    add column if not exists profile_type text,
    add column if not exists customer_id text,
    add column if not exists product_name text,
    add column if not exists min_temperature double precision,
    add column if not exists max_temperature double precision,
    add column if not exists min_humidity double precision,
    add column if not exists max_humidity double precision,
    add column if not exists target_temperature double precision,
    add column if not exists parameter_limits jsonb not null default '{}'::jsonb,
    add column if not exists research_product boolean not null default false;

create unique index if not exists profiles_cargo_profile_id_unique
    on public.profiles (profile_id)
    where profile_type = 'cargo_transport' and profile_id is not null;

create index if not exists profiles_cargo_customer_id_idx
    on public.profiles (customer_id)
    where profile_type = 'cargo_transport';
