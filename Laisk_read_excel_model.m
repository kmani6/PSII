function [species,S,rate_inds] = Laisk_read_excel_model(analysis_name)

species = {};

file = [analysis_name,'/LaiskReactions.xlsx']; 
[~,reactions,~] = xlsread(file);

nrxn = length(reactions); 

Si = [];% The row number that is being sparsed
Sj = [];% The corresponding column number that is being sparsed
Ss = [];% The matrice S

rate_inds = cell(nrxn,1); %creates rate_inds cell, a 59x1 empty cell array

for irxn = 1:nrxn 
    
    rstring = reactions{irxn}; %inds the current reaction in the rxns vector and sets as the rxn string
    sep_ind = strfind(rstring,'->'); %finds a string withing a string, and returns the idx
    lhs = strtrim(rstring(1:sep_ind-1)); %string trim b4 idx 
    reactants = strtrim(strsplit(lhs,'+')); %splits the string into 2 components (sep. reactants, removes +)
    
    %WE SHOULD BE ABLE TO REMOVE STRTRIM IN 29
    
    n = regexp(reactants,'(?<stoich>\(?[0-9.]+\)?\s+|)(?<species>\S+)', 'names');
    
    rate_inds{irxn} = []; %indices irxn in rate_inds and sets it as empty
    
    for ireactant = 1:length(n) 
        
        if isempty(find(strcmp(species,n{ireactant}.species)))
            
            species{end+1} = n{ireactant}.species;
            
        end
        
        if ~isempty(n{ireactant}.stoich)
            s = regexp(n{ireactant}.stoich, '\(?([0-9.]+)\)?', 'tokens');
            stoich = str2double(s{1});
        else
            stoich = 1;
        end
        ireactant_ind = find(strcmp(species,n{ireactant}.species));
        Si = [Si;ireactant_ind];
        Sj = [Sj;irxn];
        Ss = [Ss; -stoich];
        rate_inds{irxn} = [rate_inds{irxn}; ireactant_ind];
        
    end
    
    
    rhs = strtrim(rstring(sep_ind+2:end));
    products = strtrim(strsplit(rhs,' + '));
    n = regexp(products, '(?<stoich>\(?[0-9.]+\)?\s+|)(?<species>\S+)', 'names');
    
    for iproduct = 1:length(n)
        if ~isempty(n{iproduct})
            if isempty(find(strcmp(species,n{iproduct}.species)))
                species{end+1} = n{iproduct}.species;
            end
            if ~isempty(n{iproduct}.stoich)
                s = regexp(n{iproduct}.stoich, '\(?([0-9.]+)\)?', 'tokens');
                stoich = str2double(s{1});
            else
                stoich = 1;
            end
            
            iproduct_ind = find(strcmp(species,n{iproduct}.species));
            Si = [Si;iproduct_ind];
            Sj = [Sj;irxn];
            Ss = [Ss; stoich];
            
        end
    end
end
S = sparse(Si,Sj,Ss);

foo = 1;

