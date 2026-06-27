import os
import glob

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    fix_code = '''
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
'''
    if 'sys.stdout.reconfigure(encoding=' in content:
        return
        
    lines = content.split('\n')
    insert_idx = 0
    in_docstring = False
    for i, line in enumerate(lines):
        if line.startswith('\"\"\"') or line.startswith('\'\'\''):
            if not in_docstring:
                in_docstring = True
                if len(line) > 3 and (line.endswith('\"\"\"') or line.endswith('\'\'\'')):
                    in_docstring = False
            else:
                in_docstring = False
                insert_idx = i + 1
                break
        elif not in_docstring and not line.startswith('#'):
            if 'import' in line:
                insert_idx = i + 1
            else:
                pass
                
    # just put it after the imports
    lines.insert(insert_idx, fix_code)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

for f in glob.glob('**/*.py', recursive=True):
    fix_file(f)
