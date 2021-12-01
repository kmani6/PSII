function rs = calc_reaction_rates(ys, krxn, rate_inds)



rs = {};
rs{end+1} = [];
for i  = 2:length(ys)
    y = ys{i};
    %t = [t,ts{i}];    

    nrxn = length(rate_inds);

    r = zeros(nrxn,size(y, 2));
    for t = 1:size(y,2)
        for irxn = 1:nrxn
            r(irxn,t) = krxn(irxn)*prod(y(rate_inds{irxn},t));
        end
    end
    rs{end+1} = r;
end
end
