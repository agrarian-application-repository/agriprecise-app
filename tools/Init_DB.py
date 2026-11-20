"""Create the rtk_positions table (reference schema; the AGRARIAN database already has it).

Connection settings come from the same DB_* variables as the receiver, e.g.
`set -a; . ./.env; set +a; python tools/Init_DB.py`.
"""
import os
import psycopg2


def initialize_database():
    try:
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT", "5"))
        )
        cursor = connection.cursor()
        # anomaly: 1 anomalous, 0 normal, NULL not analysed (not part of a run of 120
        # consecutive 1 Hz epochs, or the detector gave no answer)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS rtk_positions (
            id SERIAL PRIMARY KEY,
            rover_id VARCHAR(50) NOT NULL,
            gps_week INTEGER NOT NULL,
            gps_tow DOUBLE PRECISION NOT NULL,
            latitude DOUBLE PRECISION NOT NULL,
            longitude DOUBLE PRECISION NOT NULL,
            height DOUBLE PRECISION NOT NULL,
            q_flag INTEGER NOT NULL,
            sdne DOUBLE PRECISION,
            sdeu DOUBLE PRECISION,
            sdun DOUBLE PRECISION,
            anomaly INTEGER,
            -- Makes a retried upload harmless (the receiver inserts with ON CONFLICT DO NOTHING)
            CONSTRAINT rtk_positions_rover_epoch_key UNIQUE (rover_id, gps_week, gps_tow)
        );
        """)
        connection.commit()
        print("Database initialized. Table 'rtk_positions' is ready.")
    except Exception as error:
        print(f"Error connecting to PostgreSQL: {error}")
    finally:
        if 'connection' in locals() and connection:
            cursor.close()
            connection.close()


if __name__ == "__main__":
    initialize_database()
