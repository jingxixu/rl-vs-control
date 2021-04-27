function sys = load_sys(load_dir, fnm)
    load(fullfile(load_dir, fnm), 'A', 'B', 'C', 'D')
    ts = 0.02;
    sys = ss(A, B, C, D, ts);
end