#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAF Engine - Clean Single-File Pipeline
No deps, pure Python. Steps: mask -> pyramid -> PatchMatch (Cauchy+frisket+shotgun) -> vote -> Poisson
"""

import math
import array
import random
import heapq

def _downsample(img, mask, w, h, ch):
    w2 = max(4, w//2)
    h2 = max(4, h//2)
    img2 = bytearray(w2*h2*ch)
    mask2 = bytearray(w2*h2)
    for y2 in range(h2):
        py0 = y2*2
        for x2 in range(w2):
            px0 = x2*2
            s = [0]*ch
            mv = 0
            for dy in range(2):
                sy = py0+dy
                if sy >= h: sy = h-1
                row = sy*w
                for dx in range(2):
                    sx = px0+dx
                    if sx >= w: sx = w-1
                    p = row+sx
                    if mask[p] > 10: mv = 255
                    pp = p*ch
                    for c in range(ch):
                        s[c] += img[pp+c]
            idx2 = y2*w2+x2
            pp2 = idx2*ch
            for c in range(ch):
                img2[pp2+c] = s[c]//4
            mask2[idx2] = mv
    return img2, mask2, w2, h2

def _build_sat(mask, w, h):
    W1 = w+1
    sat = array.array('i', bytes(4*((h+1)*W1)))
    for y in range(h):
        base = y*w
        srow = (y+1)*W1
        prow = y*W1
        rs = 0
        for x in range(w):
            if mask[base+x]:
                rs+=1
            sat[srow+x+1] = sat[prow+x+1] + rs
    return sat

def _rect_sum(sat, W1, x0,y0,x1,y1):
    if x0<0: x0=0
    if y0<0: y0=0
    if x1>=W1-1: x1=W1-2
    if y1>= (len(sat)//W1 -1): y1=len(sat)//W1-2
    if x1 < x0 or y1 < y0: return 0
    return sat[(y1+1)*W1+x1+1] - sat[y0*W1+x1+1] - sat[(y1+1)*W1+x0] + sat[y0*W1+x0]

def inpaint(img_bytes, mask_bytes, width, height, channels=4, patch_radius=4, quality="balanced", sample_source="auto", progress_callback=None, blend_mode="poisson", poisson_band=16, poisson_iters=40, feather_width=12, sampler_expand=1.5):
    total = width*height
    r = max(2, int(patch_radius))
    # mask analysis
    hole_pixels=[]
    band=[]
    min_x, max_x = width, 0
    min_y, max_y = height, 0
    mask = bytearray(total)
    known=[]
    for y in range(height):
        row=y*width
        for x in range(width):
            idx=row+x
            if mask_bytes[idx]>10:
                mask[idx]=1
                hole_pixels.append((x,y))
                if x<min_x: min_x=x
                if x>max_x: max_x=x
                if y<min_y: min_y=y
                if y>max_y: max_y=y
                for ndx,ndy in ((-1,0),(1,0),(0,-1),(0,1)):
                    nx=x+ndx; ny=y+ndy
                    if 0<=nx<width and 0<=ny<height and mask_bytes[ny*width+nx]<=10:
                        band.append((x,y))
                        break
            else:
                mask[idx]=0
                if r<=x<width-r and r<=y<height-r:
                    known.append((x,y))
    if not hole_pixels or not known:
        return img_bytes
    sel_w = max_x-min_x+1
    sel_h = max_y-min_y+1
    # frisket 1.5x
    fx0 = max(0, min_x - int(sel_w*sampler_expand))
    fy0 = max(0, min_y - int(sel_h*sampler_expand))
    fx1 = min(width-1, max_x + int(sel_w*sampler_expand))
    fy1 = min(height-1, max_y + int(sel_h*sampler_expand))
    # filter known to frisket
    known_f = [(x,y) for x,y in known if fx0 <= x <= fx1 and fy0 <= y <= fy1]
    if len(known_f) >= 20:
        known = known_f
    # dominant offsets voting (He & Sun) limited to frisket
    if progress_callback: progress_callback(0.06, "Voting offsets")
    hist={}
    step = max(1, len(band)//60)
    sub = band[::step]
    stride = max(6, min(width,height)//30)
    for sy in range(max(1, fy0), min(height-1, fy1), stride):
        for sx in range(max(1, fx0), min(width-1, fx1), stride):
            if mask[sy*width+sx]: continue
            for bx,by in sub:
                # need at least 5 known around band pixel
                k=0
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        nx=bx+dx; ny=by+dy
                        if 0<=nx<width and 0<=ny<height and not mask[ny*width+nx]:
                            k+=1
                if k<5: continue
                key=((sx-bx)//12, (sy-by)//12)
                ssd=0; cnt=0
                for dy in (-1,0,1):
                    by2=by+dy; sy2=sy+dy
                    if not (0<=by2<height and 0<=sy2<height): continue
                    rb=by2*width; rs=sy2*width
                    for dx in (-1,0,1):
                        bx2=bx+dx; sx2=sx+dx
                        if not (0<=bx2<width and 0<=sx2<width): continue
                        if mask[rb+bx2] or mask[rs+sx2]: continue
                        bp=(rb+bx2)*channels; sp=(rs+sx2)*channels
                        dr=img_bytes[bp]-img_bytes[sp]; dg=img_bytes[bp+1]-img_bytes[sp+1]; db=img_bytes[bp+2]-img_bytes[sp+2]
                        ssd+=dr*dr+dg*dg+db*db; cnt+=1
                if cnt==0: continue
                avg=ssd//cnt
                if avg<450:
                    hist[key]=hist.get(key,0)+(500-avg)
    dom = [(k[0]*12,k[1]*12) for k,_ in sorted(hist.items(), key=lambda kv:kv[1], reverse=True)[:6]]
    # edge priors
    pad=r+2
    edge=[]
    if min_y <= pad: edge.append((0, sel_h+pad))
    if max_y >= height-1-pad: edge.append((0, -(sel_h+pad)))
    if min_x <= pad: edge.append((sel_w+pad,0))
    if max_x >= width-1-pad: edge.append((-(sel_w+pad),0))
    if sample_source=="right": dom.insert(0,(sel_w,0))
    elif sample_source=="left": dom.insert(0,(-sel_w,0))
    elif sample_source=="above": dom.insert(0,(0,-sel_h))
    elif sample_source=="below": dom.insert(0,(0,sel_h))
    elif edge: dom = edge + dom
    else: dom.extend([(sel_w,0),(-sel_w,0),(0,sel_h),(0,-sel_h)])
    uniq=[]
    seen=set()
    for ox,oy in dom:
        if (ox,oy) not in seen and (ox!=0 or oy!=0):
            seen.add((ox,oy)); uniq.append((ox,oy))
    dom=uniq[:8]
    # pyramid sized to hole
    max_dim=max(width,height)
    if max_dim>=350 and min(width,height)>=180: n_lev=3
    elif max_dim>=160: n_lev=2
    else: n_lev=1
    levels=[(img_bytes,mask_bytes,width,height)]
    ti,tm,tw,th = img_bytes,mask_bytes,width,height
    for _ in range(n_lev-1):
        ti,tm,tw,th = _downsample(ti,tm,tw,th,channels)
        levels.append((ti,tm,tw,th))
    levels.reverse()
    pyr=[]
    for li,(lim,lam,lw,lh) in enumerate(levels):
        rem=n_lev-li-1
        rl=r if rem==0 else max(2, min(int(round(r*(0.75**rem))), min(lw,lh)//6))
        div=2**rem
        d2=[(ox//div, oy//div) for ox,oy in dom]
        pyr.append((lim,lam,lw,lh,rl,d2))
    # iterative solve
    nnf_init=None
    # helper vote
    def do_vote(nx_arr, ny_arr, d_arr, cur_img, cur_mask, W,H, R):
        vals=sorted(v for v in d_arr if v < (1<<29))
        if not vals: return
        med=float(vals[len(vals)//2])
        h_sig=max(med*0.75, 300.0)
        cutoff=med*12.0
        acc=[0.0]*(W*H*channels)
        wsum=[0.0]*(W*H)
        for x,y in hole_pixels:
            # map to current level? For vote we are at finest level only, hole_pixels is finest
            # This vote is only called for finest, so W==width etc.
            idx=y*W+x
            d=d_arr[idx]
            if d>=cutoff: continue
            wgt=math.exp(-d/h_sig)
            sx=nx_arr[idx]; sy=ny_arr[idx]
            for dy in range(-R,R+1):
                ty=y+dy
                if ty<0 or ty>=H: continue
                sy2=sy+dy
                row_t=ty*W; row_s=sy2*W
                for dx in range(-R,R+1):
                    tx=x+dx
                    if tx<0 or tx>=W: continue
                    ti=row_t+tx
                    if cur_mask[ti]==0: continue
                    si=row_s+sx+dx
                    wsum[ti]+=wgt
                    tp=ti*channels; sp=si*channels
                    acc[tp]+=wgt*cur_img[sp]
                    acc[tp+1]+=wgt*cur_img[sp+1]
                    acc[tp+2]+=wgt*cur_img[sp+2]
        for x,y in hole_pixels:
            ti=y*W+x
            ws=wsum[ti]
            if ws>1e-6:
                tp=ti*channels
                cur_img[tp]=max(0,min(255,int(acc[tp]/ws+0.5)))
                cur_img[tp+1]=max(0,min(255,int(acc[tp+1]/ws+0.5)))
                cur_img[tp+2]=max(0,min(255,int(acc[tp+2]/ws+0.5)))
                if channels==4: cur_img[tp+3]=255
    # solve coarse/mid
    for li in range(len(pyr)-1):
        lim,lam,lw,lh,rl,dl = pyr[li]
        if progress_callback: progress_callback(0.12+0.14*li, f"PatchMatch level {li+1}/{n_lev}")
        nx,ny,dist = _solve(lim,lam,lw,lh,channels,rl,dl, 5 if li==0 else 3, nnf_init, sample_source, sel_w, sel_h)
        # upsample
        nw,nh = pyr[li+1][2], pyr[li+1][3]
        sxf=nw/float(lw); syf=nh/float(lh)
        upx=array.array('i', bytes(4*nw*nh))
        upy=array.array('i', bytes(4*nw*nh))
        nmask=pyr[li+1][1]; rnxt=pyr[li+1][4]; nimg=pyr[li+1][0]
        for y in range(nh):
            yc=min(lh-1,int(y/syf)); rn=y*nw; rc=yc*lw
            for x in range(nw):
                idn=rn+x
                if nmask[idn]>10:
                    xc=min(lw-1,int(x/sxf)); icc=rc+xc
                    sxn=int(nx[icc]*sxf); syn=int(ny[icc]*syf)
                    sxn=max(rnxt,min(nw-1-rnxt,sxn)); syn=max(rnxt,min(nh-1-rnxt,syn))
                    upx[idn]=sxn; upy[idn]=syn
                    pc=icc*channels; pn=idn*channels
                    for c in range(channels): nimg[pn+c]=lim[pc+c]
                    if channels==4: nimg[pn+3]=255
                else:
                    upx[idn]=x; upy[idn]=y
        nnf_init=(upx,upy)
    # finest EM
    flim,flam,fw,fh,fr,fdom = pyr[-1]
    if progress_callback: progress_callback(0.55, "Fine synthesis")
    nx,ny,dist = _solve(flim,flam,fw,fh,channels,fr,fdom, 2, nnf_init, sample_source, sel_w, sel_h)
    if progress_callback: progress_callback(0.62, "Voting 1")
    do_vote(nx,ny,dist, flim, flam, fw, fh, fr)
    if progress_callback: progress_callback(0.74, "Refine")
    nx,ny,dist = _solve(flim,flam,fw,fh,channels,fr,fdom, 2, (nx,ny), sample_source, sel_w, sel_h)
    if progress_callback: progress_callback(0.84, "Voting 2")
    do_vote(nx,ny,dist, flim, flam, fw, fh, fr)
    # copy back if pyramid had levels (flim is img_bytes when n_lev==1 else separate)
    if flim is not img_bytes:
        # flim already is img_bytes when n_lev==1, else we need to ensure img_bytes has final
        # In multi-level case, flim is the finest level which is img_bytes reference, so already done
        pass
    # blending based on mode
    if blend_mode == "poisson":
        if progress_callback: progress_callback(0.92, "Poisson seamless blend")
        _poisson_band(img_bytes, mask, width, height, channels, hole_pixels, mask, band=poisson_band, iters=poisson_iters)
    elif blend_mode == "feather":
        if progress_callback: progress_callback(0.92, "Feather blend")
        _feather_band(img_bytes, mask, width, height, channels, hole_pixels, feather_width)
    # else none: hard copy, no blending
    return img_bytes

def _feather_band(img_bytes, mask, width, height, channels, hole_pixels, feather_width=12):
    if not hole_pixels or feather_width <= 0:
        return
    # distance to known for feather
    # simple 2-pass distance transform for feather
    total = width*height
    dist = array.array('f', [1e9]*total)
    for x,y in hole_pixels:
        # check if adjacent to known
        for ndx,ndy in ((-1,0),(1,0),(0,-1),(0,1)):
            nx=x+ndx; ny=y+ndy
            if 0<=nx<width and 0<=ny<height and mask[ny*width+nx]==0:
                dist[y*width+x]=0.0
                break
    # fast 2-pass
    for y in range(height):
        for x in range(width):
            idx=y*width+x
            if mask[idx]==1 and dist[idx]>0:
                d=dist[idx]
                if x>0: d=min(d, dist[idx-1]+1)
                if y>0: d=min(d, dist[idx-width]+1)
                dist[idx]=d
    for y in range(height-1,-1,-1):
        for x in range(width-1,-1,-1):
            idx=y*width+x
            if mask[idx]==1:
                d=dist[idx]
                if x<width-1: d=min(d, dist[idx+1]+1)
                if y<height-1: d=min(d, dist[idx+width]+1)
                dist[idx]=d
    for x,y in hole_pixels:
        idx=y*width+x
        d=dist[idx]
        if d < feather_width:
            alpha = 0.5 * (1 - math.cos(math.pi * d / feather_width))
            # blend towards average of neighboring known (approx)
            # for feather, we just keep as is but could smooth slightly
            # simple: do nothing, feather is handled by Poisson; for pure feather we blend hole edge towards known average
            # Here we do a light blend with neighboring known average
            avg_r=avg_g=avg_b=cnt=0
            for ndx,ndy in ((-1,0),(1,0),(0,-1),(0,1)):
                nx=x+ndx; ny=y+ndy
                if 0<=nx<width and 0<=ny<height and mask[ny*width+nx]==0:
                    p=(ny*width+nx)*channels
                    avg_r+=img_bytes[p]; avg_g+=img_bytes[p+1]; avg_b+=img_bytes[p+2]; cnt+=1
            if cnt>0:
                tp=idx*channels
                # blend feather: more known near edge
                w = 1 - alpha
                img_bytes[tp]=int(img_bytes[tp]*alpha + (avg_r//cnt)*w + 0.5)
                img_bytes[tp+1]=int(img_bytes[tp+1]*alpha + (avg_g//cnt)*w + 0.5)
                img_bytes[tp+2]=int(img_bytes[tp+2]*alpha + (avg_b//cnt)*w + 0.5)

def _solve(img_bytes, mask_bytes, width, height, channels, patch_radius, dominant_vecs, num_iters, initial_nnf=None, sample_source="auto", sel_w=0, sel_h=0):
    width=int(width); height=int(height); channels=int(channels)
    r=int(patch_radius)
    total=width*height
    BIG=float(1e9)
    mask=bytearray(total)
    hole=[]
    known=[]
    for y in range(height):
        row=y*width
        for x in range(width):
            idx=row+x
            if mask_bytes[idx]>10:
                mask[idx]=1; hole.append((x,y))
            else:
                mask[idx]=0
                if r<=x<width-r and r<=y<height-r:
                    known.append((x,y))
    if not hole or not known:
        z=array.array('i', bytes(4*total))
        return z,array.array('i', bytes(4*total)), z
    # SAT
    W=width; H=height; SW=W+1
    sat=array.array('i', bytes(4*((H+1)*SW)))
    for y in range(H):
        base=y*W; srow=(y+1)*SW; prow=y*SW; rs=0
        for x in range(W):
            if mask[base+x]: rs+=1
            sat[srow+x+1]=sat[prow+x+1]+rs
    def src_ok(sx,sy):
        x0=sx-r; y0=sy-r; x1=sx+r+1; y1=sy+r+1
        if x0<0 or y0<0 or x1>W or y1>H: return False
        return (sat[y1*SW+x1]-sat[y0*SW+x1]-sat[y1*SW+x0]+sat[y0*SW+x0])==0
    patch_area=(2*r+1)*(2*r+1)
    min_known=max(6,patch_area//4)
    sup_full=tuple(dy*W+dx for dy in range(-r,r+1) for dx in range(-r,r+1))
    sup_full_c=tuple(o*channels for o in sup_full)
    supports={}; supports_c={}
    for x,y in hole:
        idx=y*W+x
        offs=[]
        y0=max(0,y-r); y1=min(H-1,y+r); x0=max(0,x-r); x1=min(W-1,x+r)
        for yy in range(y0,y1+1):
            rb=yy*W
            for xx in range(x0,x1+1):
                ti=rb+xx
                if mask[ti]==0:
                    offs.append(ti-idx)
        if len(offs)>=min_known:
            t=tuple(offs); supports[idx]=t; supports_c[idx]=tuple(o*channels for o in t)
        else:
            supports[idx]=sup_full; supports_c[idx]=sup_full_c
    sup_get=supports.get; supc_get=supports_c.get
    randint=random.randint
    def pdist(tidx,sx,sy,lim):
        if not src_ok(sx,sy): return BIG
        sb=sy*W+sx
        tb=tidx*channels; sbsb=sb*channels
        IM=img_bytes
        s=0
        for off,offc in zip(sup_get(tidx,sup_full), supc_get(tidx,sup_full_c)):
            tp=tb+offc; sp=sbsb+offc
            dr=IM[tp]-IM[sp]; dg=IM[tp+1]-IM[sp+1]; db=IM[tp+2]-IM[sp+2]
            ssd=dr*dr+dg*dg+db*db
            s+= 1.0 - 1.0/(1.0+ssd/2000.0)
            if s>=lim: return s
        return s
    # init NNF
    if initial_nnf is not None and len(initial_nnf[0])==total:
        nnf_x,nnf_y=initial_nnf
    else:
        nnf_x=array.array('i', bytes(4*total)); nnf_y=array.array('i', bytes(4*total))
        for y in range(H):
            row=y*W
            for x in range(W):
                idx=row+x
                if mask[idx]==0:
                    nnf_x[idx]=x; nnf_y[idx]=y
                else:
                    placed=False
                    for dox,doy in dominant_vecs[:2]:
                        sx=x+dox; sy=y+doy
                        if src_ok(sx,sy):
                            nnf_x[idx]=sx; nnf_y[idx]=sy; placed=True; break
                    if not placed:
                        kx,ky=known[randint(0,len(known)-1)]
                        nnf_x[idx]=kx; nnf_y[idx]=ky
    for x,y in hole:
        idx=y*W+x
        sx=nnf_x[idx]; sy=nnf_y[idx]
        if 0<=sx<W and 0<=sy<H and mask_bytes[sy*W+sx]<=10:
            sp=(sy*W+sx)*channels; tp=idx*channels
            for c in range(channels): img_bytes[tp+c]=img_bytes[sp+c]
    nnf_dist=array.array('f', [0.0]*total)
    for x,y in hole:
        idx=y*W+x
        nnf_dist[idx]=pdist(idx,nnf_x[idx],nnf_y[idx],BIG)
    max_dim=max(W,H)
    rad0=min(max_dim//2, max(sel_w,sel_h)*2+48)
    fwd=sorted(hole); bwd=list(reversed(fwd))
    filt=sample_source in ("right","left","above","below")
    for it in range(num_iters):
        fwd_flag=(it%2==0)
        holes=fwd if fwd_flag else bwd
        dm=1 if fwd_flag else -1
        changed=0
        for x,y in holes:
            idx=y*W+x
            best_sx=nnf_x[idx]; best_sy=nnf_y[idx]; best_d=nnf_dist[idx]
            nx=x-dm
            if r<=nx<W-r:
                nidx=y*W+nx
                if mask[nidx]==1:
                    csx=nnf_x[nidx]+dm; csy=nnf_y[nidx]
                    ok=True
                    if filt:
                        ok=((sample_source=="right" and csx>x) or (sample_source=="left" and csx<x) or (sample_source=="below" and csy>y) or (sample_source=="above" and csy<y))
                    if ok:
                        d=pdist(idx,csx,csy,best_d)
                        if d<best_d: best_d=d; best_sx=csx; best_sy=csy
            ny=y-dm
            if r<=ny<H-r:
                nidx=ny*W+x
                if mask[nidx]==1:
                    csx=nnf_x[nidx]; csy=nnf_y[nidx]+dm
                    ok=True
                    if filt:
                        ok=((sample_source=="right" and csx>x) or (sample_source=="left" and csx<x) or (sample_source=="below" and csy>y) or (sample_source=="above" and csy<y))
                    if ok:
                        d=pdist(idx,csx,csy,best_d)
                        if d<best_d: best_d=d; best_sx=csx; best_sy=csy
            for dox,doy in dominant_vecs:
                dsx=x+dox; dsy=y+doy
                d=pdist(idx,dsx,dsy,best_d)
                if d<best_d: best_d=d; best_sx=dsx; best_sy=dsy
            if best_d>0:
                rad=rad0
                while rad>=3:
                    rx=max(r, min(W-1-r, best_sx+randint(-rad,rad)))
                    ry=max(r, min(H-1-r, best_sy+randint(-rad,rad)))
                    d=pdist(idx,rx,ry,best_d)
                    if d<best_d: best_d=d; best_sx=rx; best_sy=ry
                    rad>>=1
            if best_sx!=nnf_x[idx] or best_sy!=nnf_y[idx]:
                nnf_x[idx]=best_sx; nnf_y[idx]=best_sy; nnf_dist[idx]=best_d
                sp=(best_sy*W+best_sx)*channels; tp=idx*channels
                for c in range(channels): img_bytes[tp+c]=img_bytes[sp+c]
                changed+=1
        if changed*50 < len(hole):
            break
    return nnf_x, nnf_y, nnf_dist

def _poisson_band(img_bytes, mask, width, height, channels, hole_pixels, mask_bytes, band=16, iters=40):
    # band Poisson - seamless gradient blending
    if not hole_pixels:
        return
    # bbox
    min_x=min(p[0] for p in hole_pixels); max_x=max(p[0] for p in hole_pixels)
    min_y=min(p[1] for p in hole_pixels); max_y=max(p[1] for p in hole_pixels)
    BAND=int(band)
    x0=max(0,min_x-BAND); y0=max(0,min_y-BAND)
    x1=min(width-1,max_x+BAND); y1=min(height-1,max_y+BAND)
    # quick SAT for holes to know band? use mask
    total=width*height
    # build set for fast lookup
    hole_set=set(hole_pixels)
    # band = hole within BAND of known
    band=[]
    # use simple distance: for each hole, check if any known within BAND via brute small search (BAND<=16, bbox limited)
    for x,y in hole_pixels:
        found=False
        for dy in range(-BAND,BAND+1):
            ny=y+dy
            if ny< y0 or ny> y1: continue
            for dx in range(-BAND,BAND+1):
                nx=x+dx
                if nx< x0 or nx> x1: continue
                if 0<=nx<width and 0<=ny<height and (nx,ny) not in hole_set:
                    # known within BAND (approx Euclidean, use max)
                    if abs(dx)+abs(dy) <= BAND:
                        found=True
                        break
            if found: break
        if found:
            band.append((x,y))
    if not band:
        band=hole_pixels[:]
    # residual arrays for band only, but use full size for simplicity
    res_r=array.array('f', [0.0]*total)
    res_g=array.array('f', [0.0]*total)
    res_b=array.array('f', [0.0]*total)
    # Dirichlet for band boundary (adjacent to known)
    for bx,by in band:
        # check if adjacent to known
        adj=False
        for ndx,ndy in ((-1,0),(1,0),(0,-1),(0,1)):
            nx=bx+ndx; ny=by+ndy
            if 0<=nx<width and 0<=ny<height and (nx,ny) not in hole_set:
                adj=True
                break
        if not adj: continue
        b_idx=by*width+bx; b_pix=b_idx*channels
        for ndx,ndy in ((-1,0),(1,0),(0,-1),(0,1)):
            nx=bx+ndx; ny=by+ndy
            if 0<=nx<width and 0<=ny<height and (nx,ny) not in hole_set:
                n_pix=(ny*width+nx)*channels
                res_r[b_idx]=float(img_bytes[n_pix]-img_bytes[b_pix])
                res_g[b_idx]=float(img_bytes[n_pix+1]-img_bytes[b_pix+1])
                res_b[b_idx]=float(img_bytes[n_pix+2]-img_bytes[b_pix+2])
                break
    band_set=set(band)
    omega=1.4
    for it in range(int(iters)):
        for x,y in band:
            if (x,y) in hole_set and any((x+dx,y+dy) not in hole_set for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)) if 0<=x+dx<width and 0<=y+dy<height):
                # skip Dirichlet boundary (keep fixed)
                # check if this band pixel is boundary (adjacent to known) -> fixed
                is_bdry=False
                for ndx,ndy in ((-1,0),(1,0),(0,-1),(0,1)):
                    nx=x+ndx; ny=y+ndy
                    if 0<=nx<width and 0<=ny<height and (nx,ny) not in hole_set:
                        is_bdry=True
                        break
                if is_bdry:
                    continue
            idx=y*width+x
            sr=sg=sb=0.0; cnt=0
            for ndx,ndy in ((-1,0),(1,0),(0,-1),(0,1)):
                nx=x+ndx; ny=y+ndy
                if 0<=nx<width and 0<=ny<height:
                    nidx=ny*width+nx
                    sr+=res_r[nidx]; sg+=res_g[nidx]; sb+=res_b[nidx]; cnt+=1
            if cnt>0:
                res_r[idx]+=omega*(sr/cnt - res_r[idx])
                res_g[idx]+=omega*(sg/cnt - res_g[idx])
                res_b[idx]+=omega*(sb/cnt - res_b[idx])
    for x,y in band:
        idx=y*width+x; tp=idx*channels
        img_bytes[tp]=max(0,min(255,int(round(img_bytes[tp]+res_r[idx]))))
        img_bytes[tp+1]=max(0,min(255,int(round(img_bytes[tp+1]+res_g[idx]))))
        img_bytes[tp+2]=max(0,min(255,int(round(img_bytes[tp+2]+res_b[idx]))))
