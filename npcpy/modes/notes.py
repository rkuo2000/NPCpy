
def enter_notes_mode(npc: Any = None) -> None:
    """
    Function Description:

    Args:

    Keyword Args:
        npc : Any : The NPC object.
    Returns:
        None

    """

    npc_name = npc.name if npc else "sibiji"
    print(f"Entering notes mode (NPC: {npc_name}). Type '/nq' to exit.")

    while True:
        note = input("Enter your note (or '/nq' to quit): ").strip()

        if note.lower() == "/nq":
            break

        save_note(note, npc)

    print("Exiting notes mode.")


def save_note(note: str, db_conn, npc: Any = None) -> None:
    """
    Function Description:
        This function is used to save a note.
    Args:
        note : str : The note to save.

    Keyword Args:
        npc : Any : The NPC object.
    Returns:
        None
    """
    current_dir = os.getcwd()
    timestamp = datetime.datetime.now().isoformat()
    npc_name = npc.name if npc else "base"
    cursor = conn.cursor()

    # Create notes table if it doesn't exist
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        note TEXT,
        npc TEXT,
        directory TEXT
    )
    """
    )

    # Insert the note into the database
    cursor.execute(
        """
    INSERT INTO notes (timestamp, note, npc, directory)
    VALUES (?, ?, ?, ?)
    """,
        (timestamp, note, npc_name, current_dir),
    )

    conn.commit()

    print("Note saved to database.")
    # save the note with the current datestamp to the current working directory
    with open(f"{current_dir}/note_{timestamp}.txt", "w") as f:
        f.write(note)
