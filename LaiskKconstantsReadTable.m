function [kconst] = LaiskKconstantsReadTable(analysis_name)

file1 = [analysis_name,'/LaiskConstants.xls'];
tablek = readtable(file1);
knames = tablek.name;

file3 = [analysis_name,'/LaiskReactions.xlsx'];
[~,Rknames] = xlsread(file3);

kconst = zeros(size(Rknames,1),1);
rate_constants_reactions = Rknames(:,2);

for i = 1:length(knames)
    rate_const = knames(i);
    rate_const_idcs = find(strcmp(rate_constants_reactions, rate_const));
    if ~isempty(rate_const_idcs)
        kconst(rate_const_idcs) = i;        
    end
    
end

end