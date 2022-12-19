function [] = wsl(cmd)
    fprintf(['...' cmd '\n'])
    if ispc
        system(['bash -c "source ~/.profile &&' cmd '"']);
    elseif isunix
        system(cmd)
    end    
end