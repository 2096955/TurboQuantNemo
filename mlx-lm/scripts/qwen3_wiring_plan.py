import yaml

filename = '../.cursor/plans/apex_positioning_and_roadmap_e3aa3d67.plan.md'

with open(filename, 'r') as f:
    content = f.read()

parts = content.split('---')
if len(parts) >= 3:
    frontmatter = parts[1]
    body = '---'.join(parts[2:])
    data = yaml.safe_load(frontmatter)
    for t in data.get('todos', []):
        if t['id'] == 'qwen3-offload-wiring':
            t['status'] = 'completed'
        if t['id'] == 'qwen3-recipe':
            t['status'] = 'completed'
            
    new_frontmatter = yaml.dump(data, default_flow_style=False, sort_keys=False)
    new_content = f"---\n{new_frontmatter}---\n{body}"
    with open(filename, 'w') as f:
        f.write(new_content)
    print("Successfully updated the plan.")
else:
    print("Failed to parse frontmatter.")
