from timeit import default_timer as timer

a = range(100000)

start = timer()
print(sorted(a))
end = timer()

print(float(end - start))


start2 = timer()
print(max(a))
end2 = timer()
print(float(end2 - start2))
