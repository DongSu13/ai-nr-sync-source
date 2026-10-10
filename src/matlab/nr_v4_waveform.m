function [rx,truth,public,diagnostic]=nr_v4_waveform(record)
% Deterministic transmitter. Only rx and public may enter an online policy.
cfg=record.config;rng(record.seed,'twister');
carrier=nrCarrierConfig('NSizeGrid',20,'SubcarrierSpacing',15);
ofdm=nrOFDMInfo(carrier);fs=ofdm.SampleRate;slotLen=sum(ofdm.SymbolLengths(1:14));
public=struct('carrier',carrier,'ofdm',ofdm);
nu=cfg.nu;assert(ismember(nu,[0 1 2 3 4 6]));n=slotLen+4096;
components=zeros(n,nu);users=struct([]);refPower=0;
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
for u=1:nu
    pci=pcis(u);offset=offsets(u);cfo=cfg.cfo_max*(2*rand-1);payload=randi([0 1],24,1);
    w=nrOFDMModulate(carrier,nr_v3_grid(pci,payload),'Windowing',0);
    placed=zeros(n,1);placed(offset+(1:slotLen))=w;chSeed=record.seed*8+u;
    filterDelay=0;maxDelay=0;
    if ~strcmp(cfg.profile,'AWGN')
        ch=nrTDLChannel('DelayProfile',cfg.profile,'DelaySpread',cfg.spread_s,...
            'MaximumDopplerShift',cfg.doppler,'SampleRate',fs,'NumReceiveAntennas',1,'Seed',chSeed);
        chInfo=info(ch);filterDelay=chInfo.ChannelFilterDelay;maxDelay=ceil(max(chInfo.PathDelays)*fs);
        assert(offset+filterDelay+maxDelay+slotLen<=n && offset+filterDelay+maxDelay<=900);
        placed=ch(placed);
    end
    placed=placed*10^(powers(u)/20).*exp(1j*2*pi*cfo*(0:n-1)'/fs);components(:,u)=placed;
    active=offset+filterDelay+sum(ofdm.SymbolLengths(1:2))+(1:sum(ofdm.SymbolLengths(3:6)));
    measuredPower=mean(abs(placed(active)).^2);if u==1,refPower=measuredPower;end
    user=struct('pci',pci,'offset',offset,'cfo',cfo,'power_db',powers(u),'channel_seed',chSeed,...
        'payload',double(payload(:))','received_power',measuredPower,'filter_delay',filterDelay,...
        'max_path_delay',maxDelay,'power_window_start',active(1),'power_window_length',numel(active));
    if u==1,users=user;else,users(end+1)=user;end
end
if nu==0
    assert(isfield(cfg,'noise_reference') && cfg.noise_reference>0);nv=cfg.noise_reference*cfg.noise_scale;
else,nv=refPower/10^(cfg.snr_db/10);end
assert(isfinite(nv) && nv>0);noise=sqrt(nv/2)*(randn(n,1)+1j*randn(n,1));rx=sum(components,2)+noise;
truth=struct('users',users,'ref_power',refPower,'noise_variance',nv);
diagnostic=struct('components',components,'noise',noise);
end
