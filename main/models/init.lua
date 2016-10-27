require 'nn'
require 'cunn'
require 'cudnn'

local M = {}

function M.setup(opt, checkpoint)
    local model
    if checkpoint then
        local model_path = paths.concat(opt.resume, checkpoint.model_file)
        assert(paths.filep(model_path), 'Saved model not found: ' .. model_path)
        print('=> Resuming model from ' .. model_path)
        model = torch.load(model_path)
        print('=> Done loading')
    else
        print('=> Creating model from file: models/' .. opt.net_type .. '.lua')
        model = require('models/' .. opt.net_type)(opt)
    end

    if torch.type(model) == 'nn.DataParallelTable' then
        model = model:get(1)
    end

    cudnn.fastest = true
    cudnn.benchmark = true

    if opt.nGPU > 1 then
        print('=> Distributing over GPUs')
        local gpus = torch.range(1, opt.nGPU):totable()

        if opt.skip_gpu > 0 and opt.skip_gpu < opt.nGPU then
            table.insert(gpus, gpus[#gpus] + 1)
            table.remove(gpus, opt.skip_gpu)
        end

        local dpt = nn.DataParallelTable(1, true, true)
            :add(model, gpus)
            :threads(function()
                local cudnn = require 'cudnn'
                cudnn.fastest, cudnn.benchmark = true, true
            end)
        dpt.gradInput = nil

        model = dpt:cuda()
    end

    local criterion = nn.BCECriterion():cuda()
    return model, criterion

end

return M
