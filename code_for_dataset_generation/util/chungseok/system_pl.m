function [] = system_pl(varargin)
% system_pl(cmd1, cmd2, cmd3, ...)
cmd=varargin{1};
if nargin>1
    for i=2:nargin
       cmd = [cmd ' & ' varargin{i}]; 
    end        
end
cmd = [cmd ' & wait ']; 
system(cmd);
system('echo parallel command execution done');
end