local M = {}

function M.create(opt, split)
    local Dataset = require('datasets/' .. opt.dataset)
    return Dataset(opt, split)
end

return M
