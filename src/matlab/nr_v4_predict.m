function prediction=nr_v4_predict(model,x)
assert(size(x,2)==model.width && all(isfinite(x(:))),'Invalid predictor input');
z=(x-model.mean(:)')./model.scale(:)';
if strcmp(model.kind,'linear')
    raw=z*model.coef(:)+model.intercept;
else
    raw=repmat(model.baseline,size(z,1),1);
    for j=1:numel(model.trees)
        tree=item(model.trees,j);
        for i=1:size(z,1)
            node=item(tree,1);
            while ~node.leaf
                if z(i,node.feature+1)<=node.threshold,index=node.left+1;else,index=node.right+1;end
                node=item(tree,index);
            end
            raw(i)=raw(i)+node.value;
        end
    end
end
if strcmp(model.task,'rank')
    prediction=zeros(size(raw));positive=raw>=0;
    prediction(positive)=1./(1+exp(-raw(positive)));
    e=exp(raw(~positive));prediction(~positive)=e./(1+e);
else,prediction=raw;end
assert(all(isfinite(prediction)));
end

function value=item(array,index)
if iscell(array),value=array{index};else,value=array(index);end
end
