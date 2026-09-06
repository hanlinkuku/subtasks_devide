"""Diagnostic screen registration. Reference quadrilateral is explicit, not a label.

This utility measures image consistency; it does not infer press or read temperature.
The supplied ROI is an inspection aid, not part of the automatic annotation pipeline.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from skimage.feature import ORB,match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform,warp
import annotate


def gray(path):
    with Image.open(path) as im:return np.asarray(im.convert('L'),dtype=float)/255


def register(reference,current,roi):
    # Exclude the fixed gripper: features should belong to the moving panel region.
    maxx=min(reference.shape[1],int(np.max(roi[:,0])+65))
    maxy=min(reference.shape[0],int(np.max(roi[:,1])+60))
    minx=max(0,int(np.min(roi[:,0])-65));miny=max(0,int(np.min(roi[:,1])-60))
    detector=ORB(n_keypoints=600,fast_threshold=.04)
    detector.detect_and_extract(reference[miny:maxy,minx:maxx])
    a=detector.keypoints[:,::-1]+[minx,miny];ad=detector.descriptors.copy()
    detector.detect_and_extract(current[:min(reference.shape[0],maxy+100),:min(reference.shape[1],maxx+100)])
    b=detector.keypoints[:,::-1];bd=detector.descriptors
    pairs=match_descriptors(ad,bd,cross_check=True,max_ratio=.8)
    if len(pairs)<8:raise ValueError('Insufficient matching features')
    model,inliers=ransac((a[pairs[:,0]],b[pairs[:,1]]),ProjectiveTransform,min_samples=4,residual_threshold=2,max_trials=400,rng=42)
    if model is None or np.count_nonzero(inliers)<8:raise ValueError('Unreliable registration')
    error=float(np.median(model.residuals(a[pairs[inliers,0]],b[pairs[inliers,1]])))
    return model,{'matches':len(pairs),'inliers':int(np.count_nonzero(inliers)),'median_residual_px':error}


def run(ep,anchor,start,end,quad):
    directory=annotate.OUT/'diagnostics/screen_registration'/f'{ep}_{start}_{end}'
    directory.mkdir(parents=True,exist_ok=True)
    source=annotate.OUT/'native'/ep
    ref=gray(source/f'{anchor:06d}.jpg');roi=np.array(quad,dtype=float).reshape(4,2)
    rect=ProjectiveTransform();rect.estimate(np.array([[0,0],[199,0],[199,279],[0,279]]),roi)
    patches=[];records=[]
    for frame in range(start,end+1):
        try:
            current=gray(source/f'{frame:06d}.jpg')
            model,quality=register(ref,current,roi) if frame!=anchor else (ProjectiveTransform(),{'inliers':None,'median_residual_px':0})
            mapped=model(roi)
            if not np.all((mapped[:,0]>=0)&(mapped[:,0]<ref.shape[1])&(mapped[:,1]>=0)&(mapped[:,1]<ref.shape[0])):raise ValueError('ROI leaves image')
            transform=ProjectiveTransform();transform.estimate(np.array([[0,0],[199,0],[199,279],[0,279]]),mapped)
            patch=warp(current,inverse_map=transform,output_shape=(280,200),preserve_range=True)
            image=Image.fromarray(np.uint8(np.clip(patch*255,0,255)));image.save(directory/f'{frame:06d}.png')
            patches.append((frame,image))
            records.append({'frame':frame,'status':'registered','quality':quality,'mean_luminance':float(patch.mean()),'quad':mapped.round(3).tolist()})
        except (ValueError,RuntimeError) as error:
            records.append({'frame':frame,'status':'unreliable','reason':str(error)})
    sheet=Image.new('RGB',(1000,308*((len(patches)+4)//5)),'#eeeade');draw=ImageDraw.Draw(sheet)
    for i,(frame,image) in enumerate(patches):
        x,y=(i%5)*200,(i//5)*308;sheet.paste(image,(x,y+28));draw.text((x+8,y+8),f'FRAME {frame}',fill='black')
    sheet.save(directory/'aligned_screens.jpg',quality=95)
    annotate.save(directory/'measurements.json',{'episode':ep,'reference_frame':anchor,'reference_quad':quad,
        'roi_source':'explicit diagnostic coordinates, not automatic detection','records':records,
        'limitations':'Registration may follow changing screen features; successful registration is not proof of identical content or physical contact.'})
    print(str(directory),sum(r['status']=='registered' for r in records),'/',len(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode');p.add_argument('--anchor',type=int,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--end',type=int,required=True);p.add_argument('--quad',type=float,nargs=8,required=True);a=p.parse_args();run(a.episode,a.anchor,a.start,a.end,a.quad)
