% November 15, 2018
%================================================================%
%        Copula-Generated Random Graph Models                    %    
%================================================================%
%                                                                %
%               Get lower triangular indices                     %
%                                                                %
%================================================================%

function linds = get_lower(N)
linds = [];
  for j = 1:(N-1)
      tmp = [(1+j):N] + (j-1)*N;
      linds = [linds,tmp];
  end

end