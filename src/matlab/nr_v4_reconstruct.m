function [estimate,features]=nr_v4_reconstruct(candidate,decoded,rx,public)
% Common pilot-estimated reconstruction. No transmitted truth is accepted.
assert(decoded.crc && decoded.pci==candidate.pci,'CRC-confirmed matching decode required');
g=nr_v3_grid(candidate.pci,decoded.payload);
assert(isequal(size(g),size(candidate.channel)) && all(isfinite(candidate.channel(:))));
w=nrOFDMModulate(public.carrier,g.*candidate.channel,'Windowing',0);
ix=candidate.offset+(1:numel(w));assert(ix(1)>=1 && ix(end)<=numel(rx));
estimate=zeros(size(rx));estimate(ix)=w;
estimate=estimate.*exp(1j*2*pi*candidate.cfo*(0:numel(rx)-1)'/public.ofdm.SampleRate);
energy=max(sum(abs(rx).^2),eps);h=abs(candidate.channel(:,3:6));
features=[sum(abs(estimate).^2)/energy, sum(abs(rx-estimate).^2)/energy,...
    var(h(:))/max(mean(h(:).^2),eps)];
assert(all(isfinite(estimate)) && all(isfinite(features)));
end
