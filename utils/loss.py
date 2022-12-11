import torch
import torch.nn.functional as F

def entropy(p, prob=True, mean=True):
    if prob:
        p = F.softmax(p)
    en = -torch.sum(p * torch.log(p+1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en


def ova_loss(out_open, label):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2

    out_open = F.softmax(out_open, 1)
    # print(out_open.size())
    label_p = torch.zeros((out_open.size(0),
                           out_open.size(2))).long().cuda()
    label_range = torch.range(0, out_open.size(0) - 1).long()
    label_p[label_range, label] = 1
    label_n = 1 - label_p
    open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                    + 1e-8) * label_p, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] +
                                                1e-8) * label_n, 1)[0])
    # open_loss_neg = torch.mean(torch.mean(-torch.log(out_open[:, 0, :] +
    #                                            1e-8) * label_n, 1)[0])
    return open_loss_pos, open_loss_neg

def open_entropy(out_open, phi=0.6):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2
    out_open = F.softmax(out_open, 1)
    # print(out_open.size())
    # print(torch.max(out_open, dim=-2).values.size())
    # p = 1 - torch.softmax(torch.max(out_open, dim=-2).values, dim=1)
    entropy = torch.sum(-out_open * torch.log(out_open + 1e-8), 1)
    entropy, _ = torch.sort(entropy) # small to big
    weight = phi ** torch.arange(out_open.size(2), dtype=torch.float).cuda()
    weight = weight.repeat(out_open.size(0), 1)
    # weight = torch.gather(weight, -1, indices)
    # print("indices: ", indices[0])
    # print("sorted: ", _[0])
    # print("weight: ", weight[0])
    # print("entropy: ", entropy[0])
    # assert False
    ent_open = torch.mean(torch.mean(torch.mul(weight, entropy), 1))
    # entropy = torch.topk(entropy, k=3, largest=False, dim=1).values
    # p = torch.softmax(entropy, dim=-1)
    # print("open_class entropy: ", entropy[-1])
    # assert p.size() == entropy.size()
    # ent_open = torch.mean(torch.sum(torch.mul(p, entropy), 1))
    # print(p[0])
    # print(entropy[0])
    # print(ent_open)
    # assert False
    # print(torch.max(entropy, dim=1))
    # print(torch.softmax(entropy, dim=1))
    # max_entropy = torch.mean(torch.max(entropy, dim=1).values)
    # print(entropy)
    # print(entropy.size())
    # print(max_entropy)
    # ent_open = torch.mean(torch.mean(torch.sum(-out_open * torch.log(out_open + 1e-8), 1), 1))
    # ent_open = torch.mean(torch.mean(entropy, 1))
    # ent_open = max_entropy
    # print(ent_open)
    return ent_open
