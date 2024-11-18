import re

def vtt_to_markdown(vtt_path, md_path):
    with open(vtt_path, 'r', encoding='utf-8') as vtt_file:
        lines = vtt_file.readlines()

    text_lines = []
    for line in lines:
        # Remove lines with timestamps and formatting
        if not re.match(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}', line) and not line.startswith('WEBVTT') and not line.startswith('Kind:') and not line.startswith('Language:'):
            clean_line = re.sub(r'<.*?>', '', line).strip()
            if clean_line:
                text_lines.append(clean_line)

    # Remove duplicate lines
    unique_text_lines = list(dict.fromkeys(text_lines))

    # Concatenate all text lines
    concatenated_text = ' '.join(unique_text_lines)

    # Write to Markdown file
    with open(md_path, 'w', encoding='utf-8') as md_file:
        md_file.write(concatenated_text)

# Example usage
if __name__ == '__main__':
    vtt_to_markdown(r'data\politique\Comment échapper à la dépression ？ - Alain Soral.fr.vtt', 'output.md')