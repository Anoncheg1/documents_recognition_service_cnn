import math


class Counter:
    def __init__(self, limit):  # , multiplyer):
        self.lim: int = limit  # int(max(amounts) * multiplyer)
        print("Counter limit:", self.lim)

    def new_count(self, one_amount):
        self.c: int = 0  # done counter
        self.r: int = math.ceil(self.lim / one_amount)  # multiplyer
        # x + y = one_amount
        # x* r + y = lim
        # y = one_amount - x  # without duplicates
        # x*r + one_amount - x = lim  # with duplicates
        # x*(r - 1) = lim - one_amount
        # x = (lim - one_amount) / (r - 1)
        if self.r == 1:
            self.wd = self.lim
        else:
            self.wd = (self.lim - one_amount) / (self.r - 1)    # take duplicates
            self.wd = self.wd * self.r

    def how_many_now(self) -> int:
        diff: int = 0
        if self.c > self.wd:
            r: int = 1
        else:
            r: int = self.r
        if (self.c + r) > self.lim:
            diff = self.c + r - self.lim  # last return

        self.c += r - diff  # update counter
        return int(r - diff)


if __name__ == '__main__':  # test
    # 15 -> 20
    counter = Counter(33)
    counter.new_count(5)

    c = 0
    for x in range(5):
        r = counter.how_many_now()
        print(r)
        # assert (0 <= r <= 2)
        c += r

    print(c)
    # assert (18 <= c <= 22)
