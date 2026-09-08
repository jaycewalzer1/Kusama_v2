"""Separate repair, cutout and secondary effects within an immutable outer scope."""
from PIL import Image, ImageChops, ImageFilter


def region_masks(allowed, obj, cutout, policy, secondary=None):
    empty = Image.new('L', allowed.size)
    alpha = cutout.getchannel('A') if cutout is not None else empty
    secondary = secondary if secondary is not None else empty
    if policy is None:
        repair = allowed.copy()
        weight = repair.copy()
    else:
        if obj is None:
            raise ValueError('Tight repair requires an object mask')
        padding = policy['padding']
        padded = obj.filter(ImageFilter.MaxFilter(2 * padding + 1)) if padding else obj.copy()
        # Fully opaque pasted pixels need no generated background. Partial-alpha
        # pixels still require repair beneath them to avoid original-object ghosts.
        covered = alpha.point(lambda x: 255 if x == 255 else 0)
        repair = ImageChops.multiply(ImageChops.lighter(padded, secondary), ImageChops.invert(covered))
        repair = ImageChops.multiply(repair, allowed)
        weight = repair.copy()
        if policy['feather']:
            # Blur only the outer repair edge; keep full weight beneath old object
            # pixels and beneath the new cutout's partially transparent edges.
            blur = repair.filter(ImageFilter.GaussianBlur(policy['feather']))
            mandatory = ImageChops.multiply(obj, ImageChops.invert(covered))
            weight = ImageChops.multiply(ImageChops.lighter(blur, mandatory), repair)
    alpha_support = alpha.point(lambda x: 255 if x else 0)
    effective = ImageChops.lighter(repair, alpha_support)
    # A generative local edit must also be able to draw the target object.
    local = ImageChops.lighter(effective, obj if obj is not None else empty)
    return {'repair': repair, 'blend': weight, 'object': obj if obj is not None else empty,
            'secondary': secondary, 'effective': effective, 'local': local}
