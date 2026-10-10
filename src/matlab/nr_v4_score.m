function scores=nr_v4_score(candidates,state,policy)
scores=zeros(1,numel(candidates));
for k=1:numel(candidates)
    f=candidates(k).features;
    switch policy.rank
        case {'correlation','sss','dmrs_coherence','llr_abs'},scores(k)=f.(policy.rank);
        case 'dmrs_fit',scores(k)=-f.dmrs_fit_error;
        case 'joint_pilot',scores(k)=f.sss*sqrt(max(f.dmrs_coherence,0));
        case 'random',scores(k)=rand;
        case 'ai',scores(k)=nr_v4_predict(policy.rank_model,nr_v4_features(candidates(k),state));
        otherwise,error('Unrecognized v4 ranking policy');
    end
end
assert(all(isfinite(scores)));
end
