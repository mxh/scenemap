require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'WBCECriterion'

local DataLoader  = require 'dataloader'
local Trainer     = require 'train'
local models      = require 'models/init'
local opts        = require 'opts'
local checkpoints = require 'checkpoints'

-- parse options
local opt = opts.parse(arg)

-- configure torch
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
torch.manualSeed(opt.manual_seed)

-- configure cutorch
cutorch.manualSeedAll(opt.manual_seed)
cutorch.setDevice(opt.main_gpu)

-- load checkpoint if available
local checkpoint, optim_state = checkpoints.latest(opt)

-- load/create model
local model = models.setup(opt, checkpoint)

-- setup dataloader
local train_loader, val_loader = DataLoader.create(opt)

-- setup loss criterion
local criterion
if opt.net_type == 'semantics' then
    -- semantic segmentation uses spatial cross entropy
    criterion = cudnn.SpatialCrossEntropyCriterion():cuda()
elseif opt.net_type == 'depth' then
    -- depth is just simple MSE
    criterion = nn.MSECriterion():cuda()
else
    -- the scenemap loss is a *weighted* cross entropy because of the larg
    -- discrepancy between negative and positive cells
    local weights = require('util/binary_weights')(train_loader, opt)
    criterion = nn.WBCECriterion(weights):cuda()
end

-- setup the trainer
local trainer = Trainer(model, criterion, opt, optim_state)

-- if we only test then the validation set is run through the loaded network
if opt.test_only then
    local test_loss, test_image = trainer:test(0, val_loader)
    print(string.format(' * Results  Err %1.8f', test_loss))
    if opt.test_image ~= 'none' then
        image.save(opt.test_image, test_image)
    end
    return
end

-- this sets the starting epoch to the last epoch of the checkpoint if there
-- is one, otherwise it uses the options (default is of course 1)
local start_epoch = checkpoint and checkpoint.epoch + 1 or opt.epoch_number
local best_loss = math.huge
for epoch = start_epoch, opt.n_epochs do
    local train_loss = trainer:train(epoch, train_loader)

    -- test validation error after each epoch
    local test_loss, test_image = trainer:test(epoch, val_loader)

    -- save model if it beats all others
    if test_loss < best_loss then
        best_loss = test_loss
        print((' * Best model  Err %1.8f'):format(test_loss))
        print('Saving best model...')
        checkpoints.save(epoch, model, trainer.optim_state)
    end

    if opt.test_image ~= 'none' then
        print('=> Saving image to ' .. opt.test_image)
        image.save(opt.test_image, test_image)
    end
end

print(string.format(' * Finished loss: %1.4f', best_loss))
