% November 15, 2018
%================================================================%
%        Copula-Generated Random Graph Models                    %    
%================================================================%
%                                                                %
%               Get upper triangular indices                     %
%                                                                %
%================================================================%

function uinds = get_upper(N)
uinds = [];
  for j = 1:(N-1)
    tmp = [j:(N-1)]*N +j;
    uinds = [uinds,tmp];
  end

end