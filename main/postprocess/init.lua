local M = {}

function M.create(opt)
    local Postprocessor = require('postprocess/' .. opt.postprocess)
    return Postprocessor()
end

return M
