require('image')

function create_loss(opt)
    function classification_loss(input, target)
        pos_input      = torch.gt(input, 0.5)
        neg_input      = torch.lt(input, 0.5)

        pos_target     = torch.gt(target, 0.5)
        pos_total      = pos_target:sum()
        neg_target     = torch.lt(target, 0.5)
        neg_total      = neg_target:sum()

        true_pos       = torch.cmul(pos_input, pos_target):sum()
        true_pos_rate  = true_pos / pos_total

        false_pos      = torch.cmul(pos_input, neg_target):sum()
        false_pos_rate = false_pos / neg_total

        true_neg       = torch.cmul(neg_input, neg_target):sum()
        true_neg_rate  = true_neg / neg_total

        false_neg      = torch.cmul(neg_input, pos_target):sum()
        false_neg_rate = false_neg / pos_total

        precision      = true_pos / (true_pos + false_pos)

        accuracy       = (true_pos + true_neg) / (pos_total + neg_total)

		tpr1_count = 0
		for batch_idx=1,target:size()[1] do
			for class_idx=1,target:size()[2] do
				for row_idx=1,target:size()[3] do
					for col_idx=1,target:size()[4] do
						if target[batch_idx][class_idx][row_idx][col_idx] > 0.5 then
							start_row = math.max(row_idx - 1, 1)
							end_row = math.min(row_idx + 1, input:size()[3])
							start_col = math.max(col_idx - 1, 1)
							end_col = math.min(col_idx + 1, input:size()[4])
							if input[batch_idx][class_idx]:sub(start_row, end_row, start_col, end_col):max() >= 0.5 then
								tpr1_count = tpr1_count + 1
							end
						end
					end
				end
			end
		end

		tpr1 = tpr1_count / pos_total

		tpr2_count = 0
		for batch_idx=1,target:size()[1] do
			for class_idx=1,target:size()[2] do
				for row_idx=1,target:size()[3] do
					for col_idx=1,target:size()[4] do
						if target[batch_idx][class_idx][row_idx][col_idx] > 0.5 then
							start_row = math.max(row_idx - 2, 1)
							end_row = math.min(row_idx + 2, input:size()[3])
							start_col = math.max(col_idx - 2, 1)
							end_col = math.min(col_idx + 2, input:size()[4])
							if input[batch_idx][class_idx]:sub(start_row, end_row, start_col, end_col):max() >= 0.5 then
								tpr2_count = tpr2_count + 1
							end
						end
					end
				end
			end
		end

		tpr2 = tpr2_count / pos_total

        return {true_pos_rate, false_pos_rate, true_neg_rate, false_neg_rate, precision, accuracy, tpr1, tpr2}

    end

    function classification_loss_log(stats)
        return ('Tpr %1.4f  Tpr+1 %1.4f  Tpr+2 %1.4f  Fpr %1.4f  Acc %1.4f'):format(stats[1], stats[7], stats[8], stats[2], stats[6])
    end

    local loss = {}
    loss.compute = classification_loss
    loss.log = classification_loss_log

    return loss
end

return create_loss
