"""
Utility functions for loading and processing Marvel character data.
"""
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Optional
from ..models.character import Character


def load_characters_from_pickle(
    file_path: str | Path,
    limit: Optional[int] = None,
    min_description_length: int = 100
) -> List[Character]:
    """
    Load character data from pickle file.

    Args:
        file_path: Path to the pickle file
        limit: Optional limit on number of characters to load
        min_description_length: Minimum length of description text to include

    Returns:
        List of Character objects
    """
    # Load pickle file
    with open(file_path, 'rb') as f:
        df = pickle.load(f)

    # Filter characters with sufficient descriptions
    df_filtered = df[df['description_text'].notna()].copy()
    df_filtered = df_filtered[
        df_filtered['description_text'].str.len() >= min_description_length
    ]

    # Apply limit if specified
    if limit:
        df_filtered = df_filtered.head(limit)

    # Convert to Character objects
    characters = []
    for _, row in df_filtered.iterrows():
        try:
            # Convert pandas row to dict and handle NaN values
            row_dict = row.to_dict()
            for key, value in row_dict.items():
                if pd.isna(value):
                    row_dict[key] = None

            char = Character(
                page_id=int(row_dict['page_id']),
                name=row_dict['name'],
                urlslug=row_dict.get('urlslug'),
                ID=row_dict.get('ID'),
                ALIGN=row_dict.get('ALIGN'),
                EYE=row_dict.get('EYE'),
                HAIR=row_dict.get('HAIR'),
                SEX=row_dict.get('SEX'),
                GSM=row_dict.get('GSM'),
                ALIVE=row_dict.get('ALIVE'),
                APPEARANCES=row_dict.get('APPEARANCES'),
                **{'FIRST APPEARANCE': row_dict.get('FIRST APPEARANCE')},
                Year=row_dict.get('Year'),
                description_text=row_dict.get('description_text')
            )
            characters.append(char)
        except Exception as e:
            print(f"Warning: Failed to parse character {row.get('name')}: {e}")
            continue

    return characters


def load_characters_from_csv(
    file_path: str | Path,
    limit: Optional[int] = None,
    min_description_length: int = 100
) -> List[Character]:
    """
    Load character data from CSV file.

    Args:
        file_path: Path to the CSV file
        limit: Optional limit on number of characters to load
        min_description_length: Minimum length of description text to include

    Returns:
        List of Character objects
    """
    df = pd.read_csv(file_path)

    # Filter characters with sufficient descriptions
    df_filtered = df[df['description_text'].notna()].copy()
    df_filtered = df_filtered[
        df_filtered['description_text'].str.len() >= min_description_length
    ]

    # Apply limit if specified
    if limit:
        df_filtered = df_filtered.head(limit)

    # Convert to Character objects
    characters = []
    for _, row in df_filtered.iterrows():
        try:
            # Convert pandas row to dict and handle NaN values
            row_dict = row.to_dict()
            for key, value in row_dict.items():
                if pd.isna(value):
                    row_dict[key] = None

            char = Character(
                page_id=int(row_dict['page_id']),
                name=row_dict['name'],
                urlslug=row_dict.get('urlslug'),
                ID=row_dict.get('ID'),
                ALIGN=row_dict.get('ALIGN'),
                EYE=row_dict.get('EYE'),
                HAIR=row_dict.get('HAIR'),
                SEX=row_dict.get('SEX'),
                GSM=row_dict.get('GSM'),
                ALIVE=row_dict.get('ALIVE'),
                APPEARANCES=row_dict.get('APPEARANCES'),
                **{'FIRST APPEARANCE': row_dict.get('FIRST APPEARANCE')},
                Year=row_dict.get('Year'),
                description_text=row_dict.get('description_text')
            )
            characters.append(char)
        except Exception as e:
            print(f"Warning: Failed to parse character {row.get('name')}: {e}")
            continue

    return characters


def get_sample_characters(
    file_path: str | Path,
    character_names: List[str],
    use_pickle: bool = True
) -> List[Character]:
    """
    Get specific characters by name from the dataset.

    Args:
        file_path: Path to the data file
        character_names: List of character names to retrieve
        use_pickle: Whether to use pickle format (True) or CSV (False)

    Returns:
        List of Character objects matching the names
    """
    if use_pickle:
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = pd.read_csv(file_path)

    # Filter by character names
    df_filtered = df[df['name'].isin(character_names)].copy()

    # Convert to Character objects
    characters = []
    for _, row in df_filtered.iterrows():
        try:
            # Convert pandas row to dict and handle NaN values
            row_dict = row.to_dict()
            for key, value in row_dict.items():
                if pd.isna(value):
                    row_dict[key] = None

            char = Character(
                page_id=int(row_dict['page_id']),
                name=row_dict['name'],
                urlslug=row_dict.get('urlslug'),
                ID=row_dict.get('ID'),
                ALIGN=row_dict.get('ALIGN'),
                EYE=row_dict.get('EYE'),
                HAIR=row_dict.get('HAIR'),
                SEX=row_dict.get('SEX'),
                GSM=row_dict.get('GSM'),
                ALIVE=row_dict.get('ALIVE'),
                APPEARANCES=row_dict.get('APPEARANCES'),
                **{'FIRST APPEARANCE': row_dict.get('FIRST APPEARANCE')},
                Year=row_dict.get('Year'),
                description_text=row_dict.get('description_text')
            )
            characters.append(char)
        except Exception as e:
            print(f"Warning: Failed to parse character {row.get('name')}: {e}")
            continue

    return characters


# Alias for convenience
def load_all_characters(
    file_path: str | Path,
    use_pickle: bool = True,
    limit: Optional[int] = None,
    min_description_length: int = 100
) -> List[Character]:
    """
    Load all characters from the dataset.

    Convenience wrapper around load_characters_from_pickle/load_characters_from_csv.

    Args:
        file_path: Path to the data file
        use_pickle: Whether to use pickle format (True) or CSV (False)
        limit: Optional limit on number of characters to load
        min_description_length: Minimum length of description text to include

    Returns:
        List of Character objects
    """
    if use_pickle:
        return load_characters_from_pickle(file_path, limit, min_description_length)
    else:
        return load_characters_from_csv(file_path, limit, min_description_length)
