local M = {}

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('SceneMap Training script')
    cmd:text()
    cmd:text('Options:')
    ---------------------
    -- GENERAL OPTIONS --
    ---------------------
    cmd:option('-dataset',       'synth',     'Options: synth | sunrgbd (not yet)')
    cmd:option('-net_type',      'direct',    'Options: direct | direct_l1 | semantics | depth')
    cmd:option('-grid_size',     2,           'Options: 2 | 4 | 8 | 16 (note: only with net_type direct or direct_l1)')
    cmd:option('-n_classes',     4,           'Number of classes')
    cmd:option('-manual_seed',   0,           'Manually set RNG seed')
    cmd:option('-nGPU',          1,           'Number of GPUs to use by default')
    cmd:option('-main_gpu',      1,           'Main GPU')
    cmd:option('-skip_gpu',      2,           'If non-zero, does not use this GPU')
    cmd:option('-backend',       'cudnn',     'Options: cudnn | cunn')
    cmd:option('-cudnn',         'fastest',   'Options: fastest | default | deterministic')
    cmd:option('-train_logger',  'train.log', 'Train error log file')
    cmd:option('-val_logger',    'val.log',   'Validation log file')
    cmd:option('-test_only',     'false',     'Only validate model')
    ----------------------
    -- TRAINING OPTIONS --
    ----------------------
    cmd:option('-n_epochs',      50,         'Number of total epochs to run')
    cmd:option('-epoch_size',    32000,      'Number of iterations per epoch')
    cmd:option('-epoch_number',  1,          'Manual epoch number (useful on restarts)')
    cmd:option('-batch_size',    32,         'Mini-batch size (1 = pure stochastic)')
    cmd:option('-resume',        'none',     'Path to directory containing checkpoint')
    cmd:option('-save_interval', 5,          'Save interval in number of epochs')
    cmd:option('-test_image',    'none',     'Filename to save test result to')
    cmd:option('-postprocess',   'none',     'Postprocessor to use')
    --------------------------
    -- OPTIMIZATION OPTIONS --
    --------------------------
    cmd:option('-lr',            0.001,      'Initial learning rate')
    cmd:option('-alpha',         0.9,        'Alpha parameter for RMSProp')
    cmd:option('-epsilon',       1e-6,       'Epsilon parameter for RMSProp')
    cmd:text()
    -------------------------
    -- DATA DAEMON OPTIONS --
    -------------------------
    cmd:option('-daemon_dir',    '/home/moos/phd/proj/scenemap/render/release', 'Path to directory containing render daemon files')
    cmd:option('-buffer_size',   719,       'Default size for data buffer')
    cmd:option('-storage_dir',   '/mnt/ramdisk', 'Path to location where renderer stores output')

    local opt = cmd:parse(arg or {})

    opt.test_only = opt.test_only ~= 'false'

    return opt
end

return M


