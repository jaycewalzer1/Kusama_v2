"""The exact model-authored text admitted and delivered to FLUX."""


def compose_prompt(proposal):
    return proposal['composition'] + '\n' + proposal['render_instruction']
