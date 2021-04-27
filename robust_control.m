function robust_control(fixation, full_state, pnoise, rgb, control_penalty, amt, trial_number)

if full_state
	ob_type = 'full';
else
	ob_type = 'z_sequence';
end

load_dir = 'control_experiment_data/trial';
load_dir = append(load_dir, num2str(trial_number));
load_dir = append(load_dir, '/params/dynamic_model/');
fixstr = erase(num2str(fixation, '%.2f') ,'.');
load_fnm = append('fix_',fixstr);
load_fnm = append(load_fnm, '_obtype_');
load_fnm = append(load_fnm, ob_type);
if pnoise
	load_fnm = append(load_fnm, '_pnoise');
	if rgb
		load_fnm  = append(load_fnm, '_rgb');
	end
end
if amt < 1000000
	load_fnm = append(load_fnm, string(amt));
end

load_fnm = append(load_fnm, '.mat')


sys = load_sys(load_dir, load_fnm);
n_cont = size(sys.B, 2);
n_meas = size(sys.C, 1);

Nx = size(sys.A, 1);
eps = control_penalty;

PA = sys.A;
PB = [eye(Nx), zeros(Nx, n_meas), sys.B];
PC = [eye(Nx); zeros(1, Nx); sys.C];
PD = [zeros(Nx+1, Nx+n_meas), [zeros(Nx,1); eps]; ...
	[zeros(n_meas,Nx), eye(n_meas)], sys.D];

P = ss(PA, PB, PC, PD, 0.02);
H = ultidyn('u', [Nx+n_meas+1, Nx+n_meas+1]);
[K, CL, gamma] = musyn(feedback(P,H), n_meas, n_cont)

save_dir = 'control_experiment_data/trial';
save_dir = append(save_dir, num2str(trial_number));
save_dir = append(save_dir, '/params/hinf');
save_fnm = append('hinf_', fixstr);
save_fnm = append(save_fnm, '_obtype_', ob_type);
if pnoise
	save_fnm = append(save_fnm, '_pnoise');
	if rgb
		save_fnm = append(save_fnm, '_rgb');
	end
end
if amt < 1000000
	save_fnm = append(save_fnm, string(amt))
end
save_fnm = append(save_fnm, '.mat')
save_sys(K, save_dir, save_fnm)
