local M = {}
local NonMaxSuppressionPostprocessor = torch.class('scenemap.NonMaxSuppressionPostprocessor', M)

require 'image'
require 'nn'
require 'cunn'
require 'cudnn'

function NonMaxSuppressionPostprocessor:postprocess(input, output, target, idx)
	smooth = torch.CudaTensor(output:size()):copy(output)
	smooth[smooth:lt(0.5)] = 0
	result = torch.CudaTensor(output:size()):copy(output)
	result[result:lt(0.5)] = 0

	gauss_kernel = image.gaussian(3):cuda()
	conv = nn.SpatialConvolution(1, 1, 3, 3, 1, 1, 1, 1):cuda()
	conv.bias:fill(0)
	conv.weight:copy(gauss_kernel)

	for class_idx=1,output:size()[2] do
		smooth:narrow(2, class_idx, 1):copy(conv:forward(smooth:narrow(2, class_idx, 1)))
	end

	for batch_idx=1,output:size()[1] do
		for class_idx=1,target:size()[2] do
			for row_idx=1,output:size()[3] do
				for col_idx=1,output:size()[4] do
					start_row = math.max(row_idx - 1, 1)
					end_row = math.min(row_idx + 1, output:size()[3])
					start_col = math.max(col_idx - 1, 1)
					end_col = math.min(col_idx + 1, output:size()[4])
					if smooth[batch_idx]:sub(1, -1, start_row, end_row, start_col, end_col):max() ~= smooth[batch_idx][class_idx][row_idx][col_idx] then
						result[batch_idx][class_idx][row_idx][col_idx] = 0
					end
				end
			end
		end
	end

	return {input, result, target}
end

return M.NonMaxSuppressionPostprocessor
