function [candidates,times]=nr_v3_search(rx,carrier,ofdm)
% No scene configuration or transmitter truth is accepted by this interface.
persistent templates sssBank cacheKey
key=[ofdm.SampleRate ofdm.Nfft carrier.NSizeGrid carrier.SubcarrierSpacing];
pssStart=sum(ofdm.SymbolLengths(1:2));symbolLen=ofdm.SymbolLengths(3);
slotLen=sum(ofdm.SymbolLengths(1:14));fs=ofdm.SampleRate;maxOffset=900;
if isempty(cacheKey) || ~isequal(cacheKey,key)
    templates=zeros(symbolLen,3);sssBank=zeros(127,1008);
    for nid2=0:2
        g=zeros(240,14);ssb=zeros(240,4);ssb(nrPSSIndices)=nrPSS(nid2);g(:,3:6)=ssb;
        w=nrOFDMModulate(carrier,g,'Windowing',0);templates(:,nid2+1)=w(pssStart+(1:symbolLen));
    end
    for pci=0:1007,sssBank(:,pci+1)=nrSSS(pci);end
    cacheKey=key;
end
assert(numel(rx)>=maxOffset+slotLen);tSearch=tic;
offsets=0:maxOffset;winPower=conv(abs(rx).^2,ones(symbolLen,1),'valid');
best=zeros(maxOffset+1,3);freq=zeros(maxOffset+1,3);
for cfo=-1500:500:1500
    corrected=rx.*exp(-1j*2*pi*cfo*(0:numel(rx)-1)'/fs);
    for nid2=0:2
        t=templates(:,nid2+1);corr=conv(corrected,conj(flipud(t)),'valid');ix=pssStart+offsets+1;
        v=abs(corr(ix))./sqrt(max(winPower(ix)*sum(abs(t).^2),eps));
        v=v(:);mask=v>best(:,nid2+1);best(mask,nid2+1)=v(mask);freq(mask,nid2+1)=cfo;
    end
end
searchSeconds=toc(tSearch);featureSeconds=0;decodeSeconds=0;candidates=struct([]);
for nid2=0:2
    values=best(:,nid2+1);sidelobe=median(values);
    for peak=1:2
        tf=tic;[pss,idx]=max(values);offset=idx-1;coarse=freq(idx,nid2+1);
        values(max(1,idx-48):min(numel(values),idx+48))=0;
        corrected=rx.*exp(-1j*2*pi*coarse*(0:numel(rx)-1)'/fs);
        grid=nrOFDMDemodulate(carrier,corrected(offset+(1:slotLen)));ssb=grid(:,3:6);
        r=ssb(nrSSSIndices);pcis=nid2:3:1007;
        sc=abs(sssBank(:,pcis+1)'*r)/max(norm(r)*sqrt(127),eps);[~,ii]=sort(sc,'descend');
        featureSeconds=featureSeconds+toc(tf);
        for hypothesis=1:2
            tf=tic;pci=pcis(ii(hypothesis));sss=sc(ii(hypothesis));
            pp=sum(ssb(nrPSSIndices).*nrPSS(nid2));sp=sum(r.*nrSSS(pci));
            cfo=coarse+angle(sp*conj(pp))/(2*pi*sum(ofdm.SymbolLengths(3:4))/fs);
            corrected=rx.*exp(-1j*2*pi*cfo*(0:numel(rx)-1)'/fs);
            gridFine=nrOFDMDemodulate(carrier,corrected(offset+(1:slotLen)));ssbFine=gridFine(:,3:6);
            di=nrPBCHDMRSIndices(pci);ds=nrPBCHDMRS(pci,0);bi=nrPBCHIndices(pci);
            [hFull,nv]=nrChannelEstimate(gridFine,di+480,ds,'AveragingWindow',[9 1]);h=hFull(:,3:6);nv=max(nv,1e-8);
            [eq,csi]=nrEqualizeMMSE(ssbFine(bi),h(bi),nv);
            llr=nrPBCHDecode(eq,pci,0,nv);llr=llr.*reshape(repmat(csi',2,1),[],1);
            raw=ssbFine(di)./ds;power=mean(abs(raw).^2);
            f=struct('pss',pss,'sss',sss,'correlation',pss+sss,'peak_ratio',pss/max(sidelobe,eps),...
                'consistency',1-abs(pss-sss),'abs_cfo',abs(cfo),'dmrs_power',power,...
                'dmrs_coherence',abs(mean(raw))^2/max(power,eps),...
                'dmrs_fit_error',mean(abs(ssbFine(di)-h(di).*ds).^2)/max(mean(abs(ssbFine(di)).^2),eps),...
                'dmrs_variation',var(abs(raw))/max(power,eps),'llr_abs',mean(abs(llr)),...
                'llr_low_fraction',mean(abs(llr)<1),'estimated_noise',nv);
            assert(all(isfinite(struct2array(f))),'Nonfinite measured feature');featureSeconds=featureSeconds+toc(tf);
            td=tic;[~,err,payload]=nrBCHDecode(llr,8,4,pci);dt=toc(td);decodeSeconds=decodeSeconds+dt;
            candidate=struct('pci',pci,'offset',offset,'cfo',cfo,'features',f,'crc',err==0,...
                'payload',double(payload(:))','decode_seconds',dt);
            if isempty(candidates),candidates=candidate;else,candidates(end+1)=candidate;end
        end
    end
end
times=struct('search_seconds',searchSeconds,'features_seconds',featureSeconds,'offline_all_decode_seconds',decodeSeconds);
end
