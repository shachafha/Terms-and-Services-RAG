def clean_company_name(s: str):
    # Create a translation table that maps the characters to None
    chars_to_remove = [" ", ".", "!"]
    translation_table = str.maketrans('', '', ''.join(chars_to_remove))
    return s.translate(translation_table).lower()