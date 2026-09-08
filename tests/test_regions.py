from PIL import Image, ImageDraw, ImageChops
from kusama.regions import region_masks


def test_tight_region_preserves_nearby_background_and_repairs_antialias():
    allowed=Image.new('L',(64,64),255); obj=Image.new('L',allowed.size)
    ImageDraw.Draw(obj).rectangle((20,10,30,50),fill=255)
    cutout=Image.new('RGBA',allowed.size);ImageDraw.Draw(cutout).rectangle((23,35,27,49),fill=(0,0,255,255))
    cutout.putpixel((23,50),(0,0,255,128))
    r=region_masks(allowed,obj,cutout,{'padding':4,'feather':2})
    assert r['repair'].getpixel((24,40))==0
    assert r['repair'].getpixel((23,50))==255
    assert r['blend'].getpixel((23,50))==255
    assert r['repair'].getpixel((40,40))==0
    assert r['effective'].histogram()[255] < allowed.histogram()[255]
    assert not ImageChops.multiply(r['blend'],ImageChops.invert(r['repair'])).getbbox()


def test_secondary_effects_are_explicit():
    allowed=Image.new('L',(64,64),255);obj=Image.new('L',allowed.size);obj.putpixel((30,30),255)
    extra=Image.new('L',allowed.size);extra.putpixel((50,50),255)
    plain=region_masks(allowed,obj,None,{'padding':0,'feather':0})
    expanded=region_masks(allowed,obj,None,{'padding':0,'feather':0},extra)
    assert plain['effective'].getpixel((50,50))==0
    assert expanded['effective'].getpixel((50,50))==255
