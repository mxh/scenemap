local M = {}

local model_to_loss = {['direct'] = 'classification_loss', ['direct_l1'] = 'classification_loss', ['semantics'] = 'multi_classification_loss', ['depth'] = 'nop_loss'}

function M.setup(opt)
    print('=> Creating loss stats function from file: loss/' .. model_to_loss[opt.net_type] .. '.lua')
    local loss = require('loss/' .. model_to_loss[opt.net_type])(opt)

    return loss
end

return M
