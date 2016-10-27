local M = {}

local model_to_display = {['direct']    = 'map',
                          ['direct_l1'] = 'map',
                          ['semantics'] = 'semantics',
                          ['depth']     = 'depth'}

function M.setup(opt)
    print('=> Creating displayer from file: displays/' .. model_to_display[opt.net_type] .. '.lua')
    local display = require('displays/' .. model_to_display[opt.net_type])(opt)

    return display
end

return M
