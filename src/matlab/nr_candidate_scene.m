function scene=nr_candidate_scene(seed,smoke)
% All hypothesis features measured by one receiver, without transmitter truth.
if nargin<2,smoke=false;end
rng(seed,'twister'); carrier=nrCarrierConfig('NSizeGrid',20,'SubcarrierSpacing',15);
info=nrOFDMInfo(carrier); fs=info.SampleRate; slotLen=sum(info.SymbolLengths(1:14));
nu=randi([2 3]); if smoke,nu=1;end
profiles={'AWGN','TDL-C'}; profile=profiles{randi(2)};
snrs=[-6 0 6 12]; snr=snrs(randi(4)); spreads=[100e-9 300e-9]; ds=spreads(randi(2));
if smoke,profile='AWGN';snr=100;end
pcis=randperm(1008,nu)-1; n=slotLen+900; rx=zeros(n,1); users=struct([]); refPower=0;
for u=1:nu
    pci=pcis(u); offset=randi([64 500]); cfo=-500+1000*rand; power=-9*rand;
    if u==1,power=0;end
    if smoke,cfo=0;offset=191;end
    payload=randi([0 1],24,1); grid=buildgrid(pci,payload);
    w=nrOFDMModulate(carrier,grid,'Windowing',0);
    placed=zeros(n,1); ix=offset+(1:slotLen); placed(ix)=w;
    chSeed=seed*7+u;
    if strcmp(profile,'TDL-C')
        ch=nrTDLChannel('DelayProfile','TDL-C','DelaySpread',ds,'MaximumDopplerShift',30,...
            'SampleRate',fs,'NumReceiveAntennas',1,'Seed',chSeed);
        placed=ch(placed);
    end
    placed=placed*10^(power/20).*exp(1j*2*pi*cfo*(0:n-1)'/fs);
    rx=rx+placed;
    if u==1
        active=offset+sum(info.SymbolLengths(1:2))+(1:sum(info.SymbolLengths(3:6)));
        refPower=mean(abs(placed(active)).^2);
    end
    userRecord=struct('pci',pci,'offset',offset,'cfo',cfo,'power_db',power,...
        'channel_seed',chSeed,'payload',payload');
    if u==1,users=userRecord;else,users(end+1)=userRecord;end
end
nv=refPower/10^(snr/10); rx=rx+sqrt(nv/2)*(randn(n,1)+1j*randn(n,1));
% No ground-truth values are passed into the receiver.
[c,replicas]=search(rx,carrier,info);
for k=1:numel(c),c(k).round=0;end
scores=arrayfun(@(x)x.features.correlation,c); [~,order]=sort(scores,'descend');
accepted=order(find([c(order).crc],1));
if ~isempty(accepted) && ~smoke
    residual=rx-replicas{accepted};
    [second,~]=search(residual,carrier,info);
    for k=1:numel(second),second(k).round=1;end
    c=[c second];
end
for k=1:numel(c)
    match=find([users.pci]==c(k).pci);
    c(k).label=~isempty(match) && c(k).crc && isequal(c(k).payload,users(match).payload);
end
scene=struct('schema','nr_gain_v2','seed',seed,'channel',profile,'delay_spread_s',ds,...
    'snr_db',snr,'users',users,'candidates',c);
end

function grid=buildgrid(pci,payload)
ssb=zeros(240,4); ssb(nrPSSIndices)=nrPSS(mod(pci,3)); ssb(nrSSSIndices)=nrSSS(pci);
ssb(nrPBCHIndices(pci))=nrPBCH(nrBCH(double(payload(:)),0,0,4,0,pci),pci,0);
ssb(nrPBCHDMRSIndices(pci))=nrPBCHDMRS(pci,0);
grid=zeros(240,14);grid(:,3:6)=ssb;
end

function [candidates,replicas]=search(rx,carrier,info)
persistent templates sssBank
pssStart=sum(info.SymbolLengths(1:2)); symbolLen=info.SymbolLengths(3);
slotLen=sum(info.SymbolLengths(1:14)); fs=info.SampleRate; maxOffset=700;
if isempty(templates)
    templates=zeros(symbolLen,3);sssBank=zeros(127,1008);
    for nid2=0:2
        g=zeros(240,14);tmp=zeros(240,4);tmp(nrPSSIndices)=nrPSS(nid2);g(:,3:6)=tmp;
        w=nrOFDMModulate(carrier,g,'Windowing',0);
        templates(:,nid2+1)=w(pssStart+(1:symbolLen));
    end
    for pci=0:1007,sssBank(:,pci+1)=nrSSS(pci);end
end
offsets=0:maxOffset;winPower=conv(abs(rx).^2,ones(symbolLen,1),'valid');
best=zeros(maxOffset+1,3);freq=zeros(maxOffset+1,3);
for cfo=[-500 0 500]
    corrected=rx.*exp(-1j*2*pi*cfo*(0:numel(rx)-1)'/fs);
    for nid2=0:2
        t=templates(:,nid2+1); corr=conv(corrected,conj(flipud(t)),'valid');
        ix=pssStart+offsets+1;
        v=abs(corr(ix))./sqrt(max(winPower(ix)*sum(abs(t).^2),eps));
        v=v(:);mask=v>best(:,nid2+1);best(mask,nid2+1)=v(mask);freq(mask,nid2+1)=cfo;
    end
end
candidates=struct([]);replicas={};
for nid2=0:2
    values=best(:,nid2+1); sidelobe=median(values);
    for peak=1:2
        [pss,idx]=max(values);offset=idx-1;coarse=freq(idx,nid2+1);
        values(max(1,idx-48):min(numel(values),idx+48))=0;
        corrected=rx.*exp(-1j*2*pi*coarse*(0:numel(rx)-1)'/fs);
        grid=nrOFDMDemodulate(carrier,corrected(offset+(1:slotLen)));
        ssb=grid(:,3:6);r=ssb(nrSSSIndices);pcis=nid2:3:1007;
        sc=abs(sssBank(:,pcis+1)'*r)/max(norm(r)*sqrt(127),eps);
        [~,ii]=sort(sc,'descend');
        for hypothesis=1:2
            pci=pcis(ii(hypothesis));sss=sc(ii(hypothesis));
            pphase=sum(ssb(nrPSSIndices).*nrPSS(nid2));sphase=sum(r.*nrSSS(pci));
            deltaT=sum(info.SymbolLengths(3:4))/fs;
            fine=angle(sphase*conj(pphase))/(2*pi*deltaT);cfo=coarse+fine;
            corrected=rx.*exp(-1j*2*pi*cfo*(0:numel(rx)-1)'/fs);
            grid=nrOFDMDemodulate(carrier,corrected(offset+(1:slotLen)));ssb=grid(:,3:6);
            di=nrPBCHDMRSIndices(pci); ds=nrPBCHDMRS(pci,0);bi=nrPBCHIndices(pci);
            [hFull,nv]=nrChannelEstimate(grid,di+480,ds,'AveragingWindow',[9 1]);
            h=hFull(:,3:6);nv=max(nv,1e-8);
            [eq,csi]=nrEqualizeMMSE(ssb(bi),h(bi),nv);
            llr=nrPBCHDecode(eq,pci,0,nv);llr=llr.*reshape(repmat(csi',2,1),[],1);
            raw=ssb(di)./ds; dmrsPower=mean(abs(raw).^2);
            coherence=abs(mean(raw))^2/max(dmrsPower,eps);
            fit=mean(abs(ssb(di)-h(di).*ds).^2)/max(mean(abs(ssb(di)).^2),eps);
            f=struct('pss',pss,'sss',sss,'correlation',pss+sss,'peak_ratio',pss/max(sidelobe,eps),...
                'consistency',1-abs(pss-sss),'abs_cfo',abs(cfo),'dmrs_power',dmrsPower,...
                'dmrs_coherence',coherence,'dmrs_fit_error',fit,'dmrs_variation',var(abs(raw))/max(dmrsPower,eps),...
                'llr_abs',mean(abs(llr)),'llr_low_fraction',mean(abs(llr)<1),'estimated_noise',nv);
            % CRC/payload calculated after feature extraction, used only offline or final confirmation.
            [~,err,payload]=nrBCHDecode(llr,8,4,pci);
            c=struct('pci',pci,'offset',offset,'cfo',cfo,'features',f,'crc',err==0,'payload',double(payload(:))');
            candidates=[candidates c];
            replica=zeros(size(rx));
            if err==0
                rebuilt=buildgrid(pci,payload);rebuilt(:,3:6)=rebuilt(:,3:6).*h;
                w=nrOFDMModulate(carrier,rebuilt,'Windowing',0);ix=offset+(1:slotLen);
                replica(ix)=w.*exp(1j*2*pi*cfo*(ix(:)-1)/fs);
            end
            replicas{end+1}=replica;
        end
    end
end
end
