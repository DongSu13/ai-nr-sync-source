function run=nr_v4_control(rx,public,policy,budget,maxSearches,deps,initial)
% Truth-free online execution. Test instrumentation replaces radio primitives,
% not the state machine; production always uses the defaults below.
if nargin<6 || isempty(deps),deps=struct('search',@nr_v4_search,'decode',@nr_v4_decode,'reconstruct',@nr_v4_reconstruct);end
assert(budget>=0 && budget==fix(budget) && maxSearches>=1 && maxSearches==fix(maxSearches));
tTotal=tic;s=struct('rx',rx,'original_energy',max(sum(abs(rx).^2),eps),...
    'decode_count',0,'search_count',0,'outputs',struct([]));
events=struct([]);reconstructCount=0;decodeSeconds=0;searchSeconds=0;
featureSeconds=0;reconstructSeconds=0;scoreSeconds=0;
capture=isfield(policy,'capture') && policy.capture;rankContexts={};decisionContexts={};
if nargin>=7
    assert(isequal(rx,initial.s.rx),'Continuation waveform mismatch');
    s=initial.s;c=initial.c;tried=initial.tried;
    assert(s.decode_count<=budget && s.search_count<=maxSearches);
elseif budget>0
    [c,t]=deps.search(s.rx,public);s.search_count=1;
    searchSeconds=t.search_seconds;featureSeconds=t.features_seconds;tried=[];
else,c=struct([]);tried=[];end
while s.decode_count<budget
    ts=tic;scores=nr_v4_score(c,s,policy);scoreSeconds=scoreSeconds+toc(ts);
    [~,order]=sort(scores,'descend');chosen=[];
    for k=order
        if ~ismember(c(k).pci,tried) && (isempty(s.outputs) || ~ismember(c(k).pci,[s.outputs.pci]))
            chosen=k;break;
        end
    end
    if isempty(chosen),break;end
    if capture,rankContexts{end+1}=struct('s',s,'c',c,'tried',tried);end
    candidate=c(chosen);x=nr_v4_features(candidate,s);
    tried(end+1)=candidate.pci;s.decode_count=s.decode_count+1;
    d=deps.decode(candidate);decodeSeconds=decodeSeconds+d.seconds;
    event=struct('attempt',s.decode_count,'residual_version',s.search_count,...
        'candidate_id',candidate.id,'pci',candidate.pci,'offset',candidate.offset,...
        'cfo',candidate.cfo,'score',scores(chosen),'features',x,'crc',logical(d.crc),...
        'payload',d.payload,'cancelled',false,'gate_features',[],...
        'energy_before',sum(abs(s.rx).^2),'energy_after',sum(abs(s.rx).^2));
    if d.crc
        output=struct('pci',d.pci,'payload',d.payload,'attempt',s.decode_count,'round',s.search_count);
        if isempty(s.outputs),s.outputs=output;else,s.outputs(end+1)=output;end
        if s.decode_count<budget && s.search_count<maxSearches && ~strcmp(policy.mode,'no_sic')
            tr=tic;[estimate,g]=deps.reconstruct(candidate,d,s.rx,public);
            reconstructSeconds=reconstructSeconds+toc(tr);reconstructCount=reconstructCount+1;
            event.gate_features=[x,g,budget-s.decode_count];
            if capture && numel(decisionContexts)<4
                decisionContexts{end+1}=struct('initial',struct('s',s,'c',c,'tried',tried),...
                    'estimate',estimate,'features',event.gate_features);
            end
            switch policy.gate
                case 'always',accept=true;
                case 'energy_0.95',accept=g(2)<=.95;
                case 'energy_1',accept=g(2)<=1;
                case 'random',accept=rand>=.5;
                case 'ai',accept=nr_v4_predict(policy.gate_model,event.gate_features)>=policy.gate_threshold;
                otherwise,error('Unrecognized v4 cancellation gate');
            end
            if accept
                s.rx=s.rx-estimate;event.cancelled=true;event.energy_after=sum(abs(s.rx).^2);
                [c,t]=deps.search(s.rx,public);s.search_count=s.search_count+1;tried=[];
                searchSeconds=searchSeconds+t.search_seconds;featureSeconds=featureSeconds+t.features_seconds;
            end
        end
    end
    if isempty(events),events=event;else,events(end+1)=event;end
end
run=struct('outputs',s.outputs,'events',events,'decode_count',s.decode_count,...
    'search_count',s.search_count,'reconstruct_count',reconstructCount,...
    'budget',budget,'max_searches',maxSearches,'policy',policy,...
    'timings',struct('total_seconds',toc(tTotal),'search_seconds',searchSeconds,...
    'features_seconds',featureSeconds,'decode_seconds',decodeSeconds,...
    'reconstruct_seconds',reconstructSeconds,'score_seconds',scoreSeconds));
if capture,run.rank_contexts=rankContexts;run.decision_contexts=decisionContexts;end
end
