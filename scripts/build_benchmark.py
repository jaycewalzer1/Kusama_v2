"""Rebuild annotated synthetic calibration fixtures and explicit benchmark plans."""
from pathlib import Path
import json
from PIL import Image, ImageDraw
from kusama.artifacts import sha256

root = Path(__file__).resolve().parents[1]
out = root/'examples/benchmark'
out.mkdir(exist_ok=True)
cases = []

def save_plan(name, data):
    (out/f'{name}.json').write_text(json.dumps(data, indent=2)+'\n')
    cases.append({'id': name, 'plan': f'{name}.json'})

mountain = json.loads((root/'examples/controlled/mountain.json').read_text())
mountain.update(object_mask='../controlled/person_mask.png', allowed_mask='../controlled/allowed_mask.png',
                verification={'observer':'grounding_dino','label':'person','operation':'resize',
                              'height_tolerance': .2,'position_tolerance':6})
for name, method, tight in [('mountain_broad','geometric',False), ('mountain_tight','geometric',True), ('mountain_masked','masked',True)]:
    data = dict(mountain, method=method)
    if tight: data['repair_policy'] = {'padding':4,'feather':2}
    save_plan(name,data)

# Flat, annotated scenes intentionally isolate measurable editing failures.
for scene, color, box, shape in [('circle',(210,35,35),(95,100,157,162),'ellipse'),
                                 ('poster',(30,170,65),(91,77,165,177),'rectangle')]:
    bg=(235,228,210)
    parent=Image.new('RGB',(256,256),bg); draw=ImageDraw.Draw(parent)
    draw.rectangle((14,14,62,57),fill=(90,110,130))
    draw.line((0,207,255,207),fill=(120,110,95),width=3)
    draw.rectangle((198,155,226,202),fill=(230,190,45))
    getattr(draw,shape)(box,fill=color)
    parent.save(out/f'{scene}_parent.png')
    obj=Image.new('L',parent.size); getattr(ImageDraw.Draw(obj),shape)(box,fill=255);obj.save(out/f'{scene}_object.png')
    allowed=Image.new('L',parent.size);ImageDraw.Draw(allowed).rectangle((72,64,184,192),fill=255);allowed.save(out/f'{scene}_allowed.png')
    spec={'observer':'color','operation':'remove' if scene=='circle' else 'recolor',
          'source_color':'red' if scene=='circle' else 'green'}
    if scene=='circle': spec['background_rgb']=list(bg)
    else: spec['target_color']='blue'
    instruction=('Remove the red circle completely. Fill its former location with the same flat warm cream background. Preserve all other shapes and lines.' if scene=='circle' else
                 'Change only the green rectangle to saturated blue. Keep its exact rectangular shape, size and position, and preserve the flat cream background and all other shapes.')
    base=dict(parent=f'{scene}_parent.png',parent_sha256=sha256(out/f'{scene}_parent.png'),instruction=instruction,
              target='red circle' if scene=='circle' else 'green rectangle',allowed_mask=f'{scene}_allowed.png',
              object_mask=f'{scene}_object.png',method='masked',seed=23,max_calls=1,
              visual_checks=['Did the requested change occur without extra shapes, remnants or artifacts?',
                             'Do repaired boundaries look natural and preserve the simple illustration style?'],
              verification=spec,author='Procedural calibration fixture with known annotations; not a natural-image benchmark')
    for tight in (False,True):
        data=dict(base)
        if tight: data['repair_policy']={'padding':4,'feather':2}
        save_plan(scene+('_tight' if tight else '_broad'),data)
(out/'suite.json').write_text(json.dumps({'name':'Controlled edit verification and repair pilot','max_calls':7,'cases':cases},indent=2)+'\n')
print(out/'suite.json')
