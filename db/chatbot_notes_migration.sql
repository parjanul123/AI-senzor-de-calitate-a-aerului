-- Persistent memory for the chatbot: stores facts/instructions the user teaches it
-- (e.g. "daca senzorul de CO2 se deconecteaza, verifica alimentarea si cablul I2C"),
-- so they can be recalled automatically in future conversations.
-- Run in Supabase SQL Editor. If this table does not exist, the app falls back to a
-- local JSON file (data/chatbot_notes.json), which does not survive redeploys.

begin;

create table if not exists public.chatbot_notes (
    id bigint generated always as identity primary key,
    note text not null,
    device_identifier text,
    created_at timestamptz not null default now()
);

create index if not exists chatbot_notes_device_identifier_idx
    on public.chatbot_notes (device_identifier);

commit;
