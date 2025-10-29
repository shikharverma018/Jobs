def swap_first_two_digits(N):
    count=0
    temp=N

    while temp>0:
        temp//=10
        count+=1

    first =  N//(10**(count-1))
    second = N//(10**(count-2))
    second %=10

    last = N%(10**(count-2))

    result = second*(10**(count-1))+ first*(10**(count-2))+last
    return result
