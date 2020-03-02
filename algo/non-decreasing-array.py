class Solution:
    def checkPossibility(self, nums) -> bool:
        l = len(nums)
        i = l-1
        count = 0
        last_i = i
        while i>0:
            while i>0 and nums[i]>=nums[i-1]: i-=1
            print(i)
            if i==0 and count<=1: return True
            if i<l-1:
                nums[i] = nums[i+1]
            print(nums)
            i -= 1
            count += 1
        return count<=1
s = Solution()
assert not s.checkPossibility([3,4,2,3])
assert s.checkPossibility([2,3,3,2,4])
assert s.checkPossibility([4,2,3])
assert not s.checkPossibility([4,2,1])
        
