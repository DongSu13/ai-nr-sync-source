function scene=nr_v3_scene(record)
% Configurable first-pass experiment. The receiver receives only waveform.
cfg=record.config;rng(record.seed,'twister');
carrier=nrCarrierConfig('NSizeGrid',20,'SubcarrierSpacing',15);
info=nrOFDMInfo(carrier);fs=info.SampleRate;slotLen=sum(info.SymbolLengths(1:14));
nu=cfg.nu;assert(ismember(nu,[0 1 2 3 4 6]));
% 4096 sample tail exceeds max configured TDL path/filter delay and offset.
n=slotLen+4096;rx=zeros(n,1);users=struct([]);refPower=0;
if strcmp(cfg.pss_mode,'same')
    nid2=randi([0 2]);pcis=3*(randperm(336,nu)-1)+nid2;
elseif strcmp(cfg.pss_mode,'different')
    assert(nu<=3);pcis=3*(randi(336,1,nu)-1)+randperm(3,nu)-1;
else,pcis=randperm(1008,nu)-1;end
if strcmp(cfg.timing_mode,'near')
    base=randi([64 492]);offsets=base+randi([0 8],1,nu);
elseif strcmp(cfg.timing_mode,'separated')
    offsets=sort(randperm(437-64*max(nu-1,0),nu)+63)+(0:nu-1)*64;
else,offsets=randi([64 500],1,nu);end
powers=-cfg.gap_db*rand(1,nu);if nu>0,powers(1)=0;end;if nu>1,powers(end)=-cfg.gap_db;end
tWave=tic;
for u=1:nu
    pci=pcis(u);offset=offsets(u);cfo=cfg.cfo_max*(2*rand-1);payload=randi([0 1],24,1);
    w=nrOFDMModulate(carrier,nr_v3_grid(pci,payload),'Windowing',0);
    placed=zeros(n,1);placed(offset+(1:slotLen))=w;chSeed=record.seed*8+u;
    filterDelay=0;maxDelay=0;
    if ~strcmp(cfg.profile,'AWGN')
        ch=nrTDLChannel('DelayProfile',cfg.profile,'DelaySpread',cfg.spread_s,...
            'MaximumDopplerShift',cfg.doppler,'SampleRate',fs,'NumReceiveAntennas',1,'Seed',chSeed);
        chInfo=infoChannel(ch);filterDelay=chInfo.ChannelFilterDelay;maxDelay=ceil(max(chInfo.PathDelays)*fs);
        assert(offset+filterDelay+maxDelay+slotLen<=n,'Channel tail exceeds buffer');
        assert(offset+filterDelay+maxDelay<=900,'Channel exceeds timing search');
        placed=ch(placed);
    end
    placed=placed*10^(powers(u)/20).*exp(1j*2*pi*cfo*(0:n-1)'/fs);rx=rx+placed;
    % Reference is the SSB-duration output window shifted by channel filter delay.
    active=offset+filterDelay+sum(info.SymbolLengths(1:2))+(1:sum(info.SymbolLengths(3:6)));
    assert(active(1)>=1 && active(end)<=n,'Reference power window exceeds waveform');
    measuredPower=mean(abs(placed(active)).^2);if u==1,refPower=measuredPower;end
    user=struct('pci',pci,'offset',offset,'cfo',cfo,'power_db',powers(u),'channel_seed',chSeed,...
        'payload',double(payload(:))','received_power',measuredPower,'filter_delay',filterDelay,'max_path_delay',maxDelay,...
        'power_window_start',active(1),'power_window_length',numel(active));
    if u==1,users=user;else,users(end+1)=user;end
end
if nu==0
    assert(isfield(cfg,'noise_reference') && cfg.noise_reference>0);nv=cfg.noise_reference*cfg.noise_scale;
else,nv=refPower/10^(cfg.snr_db/10);end
assert(isfinite(nv) && nv>0);rx=rx+sqrt(nv/2)*(randn(n,1)+1j*randn(n,1));waveSeconds=toc(tWave);
[c,times]=nr_v3_search(rx,carrier,info);
for k=1:numel(c)
    match=[];if nu>0,match=find([users.pci]==c(k).pci);end
    c(k).label=~isempty(match) && c(k).crc && isequal(c(k).payload,users(match).payload);
end
scene=record;scene.schema='nr_gain_v3';scene.users=users;scene.candidates=c;
scene.ref_power=refPower;scene.noise_variance=nv;scene.received_power=mean(abs(rx).^2);
scene.timings=times;scene.timings.waveform_seconds=waveSeconds;
end

function out=infoChannel(channel)
out=info(channel);
end
