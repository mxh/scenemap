local optim = require 'optim'

local M = {}
local Trainer = torch.class('scenemap.Trainer', M)

function Trainer:__init(model, criterion, opt, optim_state)
    self.model = model
    self.criterion = criterion
    self.optim_state = optim_state or {
        learningRate = opt.lr,
        epsilon      = opt.epsilon,
        alpha        = opt.alpha
    }
    self.opt = opt

    self.train_logger = opt.train_logger ~= 'none' and optim.Logger(opt.train_logger) or nil
    self.val_logger   = opt.val_logger   ~= 'none' and optim.Logger(opt.val_logger)   or nil

    self.display = require('displays/init').setup(opt)
    self.test_image = opt.test_image

    if opt.postprocess ~= 'none' then
        self.postprocessor = require('postprocess/init').create(opt)
    end

    -- loss function, solely for performance testing purposes.
    -- this function is *not* used in any gradient descent step.
    self.lf = require('loss/init').setup(opt)

    self.params, self.gradParams = model:getParameters()
    print('Number of parameters: ' .. self.params:size(1))
end

function Trainer:train(epoch, dataloader)
    -- Train the model for a single epoch
    self.optim_state.learning_rate = self:learning_rate(epoch)

    local timer = torch.Timer()
    local data_timer = torch.Timer()

    local function feval()
        return self.criterion.output, self.gradParams
    end

    local train_size = dataloader:size()
    local loss_sum = 0.0
    local N = 0

    print('=> Training epoch # ' .. epoch)
    -- set the batch norm to training mode
    self.model:training()
    for n, sample in dataloader:run() do
        local data_time = data_timer:time().real

        -- copy input and target to gpu
        self:copy_inputs(sample)

        -- run data through network
        local output = self.model:forward(self.input):float()
        
        -- compute criterion loss
        local loss = self.criterion:forward(self.model.output, self.target)

        -- zero out any gradient params
        self.model:zeroGradParameters()
        
        -- backpropagate
        self.criterion:backward(self.model.output, self.target)
        self.model:backward(self.input, self.criterion.gradInput)

        -- update weights with rmsprop
        optim.rmsprop(feval, self.params, self.optimState)

        -- postprocess
        if self.postprocessor then
            self.input, self.model.output, self.target = table.unpack(self.postprocessor:postprocess(self.input, output, self.target, N))
        end

        -- compute stats loss (note - just for logging/statistics, this is not used in the optimization)
        local stats = self.lf.compute(self.model.output, self.target)

        loss_sum = loss_sum + loss
        N = N + 1

        print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  %s  LR %1.8f'):format(
            epoch, n, train_size, timer:time().real, data_time, loss, self.lf.log(stats), self.optim_state.learning_rate))
        
        if self.train_logger then 
            self.train_logger:add{['epoch'] = epoch, ['iteration'] = n, ['loss'] = loss}
        end

        timer:reset()
        data_timer:reset()
    end

    return loss_sum / N
end

function Trainer:test(epoch, dataloader)
    local timer = torch.Timer()
    local data_timer = torch.Timer()
    local size = dataloader:size()

    local loss_sum = 0.0
    local N = 0

    self.model:evaluate()
    for n, sample in dataloader:run() do
        local data_time = data_timer:time().real

        -- copy input and target to gpu
        self:copy_inputs(sample)

        -- run data through network
        local output = self.model:forward(self.input):float()

        -- compute criterion loss
        local loss   = self.criterion:forward(self.model.output, self.target)

        -- postprocess
        if self.postprocessor then
            self.input, self.model.output, self.target = table.unpack(self.postprocessor:postprocess(self.input, output, self.target, N))
        end

        -- compute stats loss (note - just for logging/statistics, this is not used in the optimization)
        local stats = self.lf.compute(self.model.output, self.target)

        loss_sum = loss_sum + loss
        N = N + 1

        print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f (%1.4f)  %s'):format(
            epoch, n, size, timer:time().real, data_time, loss, loss_sum / N, self.lf.log(stats)))

        timer:reset()
        data_timer:reset()
    end
    -- default mode is training, reset
    self.model:training()

    if self.val_logger then self.val_logger:add{['epoch'] = epoch, ['loss'] = loss_sum / N} end

    if self.test_image ~= 'none' then
        test_image = self.display(self.input, self.target, self.model.output)
    else
        test_image = nil
    end

    print((' * Finished epoch # %d    Err: %1.4f\n'):format(
        epoch, loss_sum / N))

    return loss_sum / N, test_image
end

function Trainer:copy_inputs(sample)
    self.input = self.input or (self.opt.nGPU == 1
        and torch.CudaTensor()
        or cutorch.createCudaHostTensor())

    self.target = self.target or torch.CudaTensor()

    self.input:resize(sample.input:size()):copy(sample.input)
    self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learning_rate(epoch)
    local decay = math.floor((epoch - 1) / 5)
    return self.opt.lr * math.pow(0.8, decay)
end

return M.Trainer
