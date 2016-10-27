require 'image'

function colour_labels(labels)
  local h = labels:size(2)
  local w = labels:size(3)
  local colouredLabels = torch.Tensor(3,h,w)
  for i=1,h do
    for j=1,w do
      l = labels[1][i][j]
      if l == 1 then
        colouredLabels[1][i][j] = 0.4157
        colouredLabels[2][i][j] = 0.5333
        colouredLabels[3][i][j] = 0.8000
      elseif l == 2 then
        colouredLabels[1][i][j] = 0.9137
        colouredLabels[2][i][j] = 0.3490
        colouredLabels[3][i][j] = 0.1882
      elseif l == 3 then
        colouredLabels[1][i][j] = 1
        colouredLabels[2][i][j] = 0.5
        colouredLabels[3][i][j] = 0
      elseif l == 4 then
        colouredLabels[1][i][j] = 1
        colouredLabels[2][i][j] = 0.8078
        colouredLabels[3][i][j] = 0.8078
      elseif l == 5 then
        colouredLabels[1][i][j] = 0.9412
        colouredLabels[2][i][j] = 0.1373
        colouredLabels[3][i][j] = 0.9216
      elseif l == 6 then
        colouredLabels[1][i][j] = 0
        colouredLabels[2][i][j] = 0.8549
        colouredLabels[3][i][j] = 0
      elseif l == 7 then
        colouredLabels[1][i][j] = 0.4588
        colouredLabels[2][i][j] = 0.1137
        colouredLabels[3][i][j] = 0.1608
      elseif l == 8 then
        colouredLabels[1][i][j] = 0.4157
        colouredLabels[2][i][j] = 0.5333
        colouredLabels[3][i][j] = 0
      elseif l == 9 then
        colouredLabels[1][i][j] = 0.5843
        colouredLabels[2][i][j] = 0
        colouredLabels[3][i][j] = 0.9412
      elseif l == 10 then
        colouredLabels[1][i][j] = 0.8706
        colouredLabels[2][i][j] = 0.9451
        colouredLabels[3][i][j] = 0.0941
      elseif l == 11 then
        colouredLabels[1][i][j] = 0
        colouredLabels[2][i][j] = 0.6549
        colouredLabels[3][i][j] = 0.6118
      elseif l == 12 then
        colouredLabels[1][i][j] = 0
        colouredLabels[2][i][j] = 0
        colouredLabels[3][i][j] = 0
      end
    end
  end
  return colouredLabels
end
