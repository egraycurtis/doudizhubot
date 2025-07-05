import psycopg2

def setup_database():
    conn = psycopg2.connect("postgresql://postgres:password@localhost:5432")  # Update connection details
    cursor = conn.cursor()
    
    # cursor.execute('''
    #     create table game_turns (
    #         id bigint generated always as identity primary key,
    #         turn_data jsonb
    #     )
    # ''')
    cursor.execute('''
        create table if not exists competitions (
            id bigint generated always as identity primary key,
            created_at timestamptz not null default now(),
            results jsonb
        )
    ''')
    conn.commit()
    conn.close()