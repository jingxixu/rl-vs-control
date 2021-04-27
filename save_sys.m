function save_sys(sys, save_dir, fnm)
    if ~isfolder(save_dir)
        mkdir(save_dir);
    end
    A = sys.A;
    B = sys.B;
    C = sys.C;
    D = sys.D;
    Nx = size(A, 1);
    save(fullfile(save_dir, fnm), 'A', 'B', 'C', 'D', 'Nx');
end