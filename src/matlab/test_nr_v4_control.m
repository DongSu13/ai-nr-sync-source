function test_nr_v4_control
assert(exist('nr_v4_control','file')==2,'v4 controller not implemented');
% Actual controller, instrument only expensive radio primitives. Truth is
% absent from every policy input. These fixtures expose budget/CRC mistakes.
calls=0;searchCalls=0;allFail=false;
deps=struct('search',@searchFixture,'decode',@decodeFixture,'reconstruct',@reconstructFixture);
policy=struct('rank','dmrs_fit','gate','always','mode','sic');
rx=ones(16,1);public=struct;
r=nr_v4_control(rx,public,policy,2,4,deps);
assert(calls==2 && r.decode_count==2 && searchCalls==1 && r.search_count==1);
assert(numel(r.outputs)==1 && r.outputs(1).pci==2);
assert(~r.events(1).cancelled && r.events(2).crc && ~r.events(2).cancelled,...
    'Final exhausted attempt must not waste cancellation/search');
calls=0;searchCalls=0;r=nr_v4_control(rx,public,policy,3,4,deps);
assert(calls==3 && searchCalls==2 && r.events(2).cancelled);
assert(r.events(3).pci==1,'Failed PCI must be retryable on new residual');
assert(numel(r.outputs)==1,'CRC output must not be duplicated');
calls=0;searchCalls=0;r=nr_v4_control(rx,public,policy,3,1,deps);
assert(searchCalls==1 && ~any([r.events.cancelled]) && numel(r.outputs)==2);
policy.gate='energy_0.95';calls=0;searchCalls=0;
r=nr_v4_control(rx,public,policy,6,4,deps);
assert(calls==3 && searchCalls==1 && numel(r.outputs)==2);
assert(~any([r.events.cancelled]),'Energy gate must reject harmful proposal');
allFail=true;calls=0;searchCalls=0;r=nr_v4_control(rx,public,policy,6,4,deps);
assert(calls==3 && isempty(r.outputs) && ~any([r.events.cancelled]));
allFail=false;policy.gate='always';policy.mode='no_sic';calls=0;searchCalls=0;
r=nr_v4_control(rx,public,policy,6,4,deps);
assert(searchCalls==1 && calls==3 && numel(r.outputs)==2 && r.reconstruct_count==0);
policy.mode='sic';policy.capture=true;calls=0;searchCalls=0;
r=nr_v4_control(rx,public,policy,6,4,deps);
assert(isfield(r,'decision_contexts'),'Missing cloneable branch decision state');
context=r.decision_contexts{1};before=context.initial.s.decode_count;
assert(before==2 && numel(context.initial.s.outputs)==1);
teacher=rmfield(policy,'capture');teacher.mode='no_sic';
skip=nr_v4_control(context.initial.s.rx,public,teacher,before+2,4,deps,context.initial);
assert(skip.decode_count-before==1 && skip.search_count==1);
assert(numel(skip.outputs)==2 && skip.outputs(1).pci==2 && skip.outputs(2).pci==3);
assert(skip.events(1).attempt==3,'Continuation reset absolute budget');
assert(numel(r.rank_contexts)==r.decode_count);
truth=struct('users',[struct('pci',2,'payload',zeros(1,24),'power_db',0),...
    struct('pci',3,'payload',zeros(1,24),'power_db',-6)]);
branch=nr_v4_branch(context,public,truth,6,4,deps);
assert(branch.benefit==-1 && branch.skip_new_users==1 && branch.cancel_new_users==0);
assert(branch.skip_decode_calls<=2 && branch.cancel_decode_calls<=2);
policy=struct('rank','ai','gate','ai','mode','sic','gate_threshold',0);
policy.rank_model=struct('kind','linear','task','rank','width',17,'mean',zeros(1,17),...
    'scale',ones(1,17),'coef',zeros(1,17),'intercept',0);
policy.gate_model=struct('kind','linear','task','gate','width',21,'mean',zeros(1,21),...
    'scale',ones(1,21),'coef',zeros(1,21),'intercept',-1);
calls=0;searchCalls=0;r=nr_v4_control(rx,public,policy,6,4,deps);
assert(calls==3 && searchCalls==1 && ~any([r.events.cancelled]) && numel(r.outputs)==2);
policy.gate_model.intercept=1;calls=0;searchCalls=0;
r=nr_v4_control(rx,public,policy,3,4,deps);
assert(r.events(2).cancelled && searchCalls==2);
fprintf('NR V4 CONTROLLER TESTS PASS\n');

    function [c,t]=searchFixture(wave,~)
        searchCalls=searchCalls+1;c=struct([]);
        ids=[1 2 2 3];fits=[.1 .2 .15 .3];
        for j=1:4
            f=struct('pss',.5,'sss',.4,'correlation',.9,'peak_ratio',2,...
                'consistency',.9,'abs_cfo',1,'dmrs_power',1,'dmrs_coherence',.5,...
                'dmrs_fit_error',fits(j),'dmrs_variation',.2,'llr_abs',2,...
                'llr_low_fraction',.1,'estimated_noise',.1);
            entry=struct('id',j,'pci',ids(j),'offset',0,'cfo',0,'features',f,...
                'decodable',ids(j)~=1 && ~(ids(j)==3 && norm(wave)<4));
            if j==1,c=entry;else,c(end+1)=entry;end
        end
        t=struct('search_seconds',0,'features_seconds',0);
    end
    function d=decodeFixture(c)
        calls=calls+1;d=struct('pci',c.pci,'crc',~allFail && c.decodable,...
            'payload',zeros(1,24),'seconds',0);
    end
    function [w,f]=reconstructFixture(~,d,wave,~)
        assert(d.crc);w=.1*wave;f=[.01 1.1 .1];
    end
end
