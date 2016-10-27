require 'image'

function visualize_map_diff(target, output)
    assert(target:dim() == 2, "only single map supported")
    assert(output:dim() == 2, "only single map supported")

    vis = torch.CudaTensor(3, target:size()[1], target:size()[2]):zero()

    outputThresholded = torch.gt(output, 0.5)

    -- start with all classifications as "good", so green
    --vis[2][torch.eq(target, 1)] =     output[torch.eq(target, 1)]
    --vis[1][torch.eq(target, 1)] = 1 - output[torch.eq(target, 1)]

    --vis[1][torch.eq(target, 0)] =     output[torch.eq(target, 0)]
    --vis[2][torch.eq(target, 0)] =     output[torch.eq(target, 0)]
    
    -- start with all classifications as "good", so green
    vis[2][torch.eq(outputThresholded, 1)] = 1
    -- set false negatives red
    vis[1][torch.gt(target, outputThresholded)] = 1
    vis[2][torch.gt(target, outputThresholded)] = 0
    -- set false positives yellow
    vis[1][torch.lt(target, outputThresholded)] = 1
    vis[2][torch.lt(target, outputThresholded)] = 1

    return vis:float()
end

function create_display(opt)
    function display(input, target, output)
        local batch_size = input:size()[1]
        local n_class = target:size()[2]
        assert(n_class == output:size()[2], "output and target don't have same number of classes")

        local ims = {}
        for batch_idx=1, batch_size do
            table.insert(ims, input[batch_idx])
            for class_idx=1, n_class do
                table.insert(ims, image.scale(visualize_map_diff(target[batch_idx][class_idx], output[batch_idx][class_idx]), input:size()[input:dim()-1], input:size()[input:dim()], 'simple'))
            end
        end

        return image.toDisplayTensor{input=ims, nrow=(1 + n_class), padding=10}
    end

    return display
end

return create_display
