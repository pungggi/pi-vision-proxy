Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public class WinCred2 {
    [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    public static extern bool CredRead(string target, int type, int reserved, out IntPtr cred);
    [DllImport("advapi32.dll")]
    public static extern void CredFree(IntPtr buf);
    [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    public static extern bool CredReadW([MarshalAs(UnmanagedType.LPWStr)] string target, int type, int reserved, out IntPtr cred);

    public static string ReadCredential(string target) {
        IntPtr credPtr = IntPtr.Zero;
        try {
            if (!CredRead(target, 1, 0, out credPtr)) return null;

            // CREDENTIAL structure offsets (64-bit):
            // Flags: 4, Type: 4, TargetName: 8, Comment: 8, LastWritten: 8,
            // CredentialBlobSize: 4, CredentialBlob: 8+4(padding)
            int blobSizeOffset = 4 + 4 + 8 + 8 + 8; // = 32
            int blobPtrOffset = blobSizeOffset + 4 + 4; // = 40 (with padding)

            int blobSize = Marshal.ReadInt32(credPtr, blobSizeOffset);
            IntPtr blobPtr = Marshal.ReadIntPtr(credPtr, blobPtrOffset);

            if (blobSize == 0 || blobPtr == IntPtr.Zero) return null;

            byte[] bytes = new byte[blobSize];
            Marshal.Copy(blobPtr, bytes, 0, blobSize);
            return Encoding.Unicode.GetString(bytes);
        } finally {
            if (credPtr != IntPtr.Zero) CredFree(credPtr);
        }
    }
}
"@

$result = [WinCred2]::ReadCredential('pi-edgee-proxy')
if ($result) {
    Write-Host "Key found. Starts with: $($result.Substring(0, [Math]::Min(15, $result.Length)))..."
    Write-Host "Length: $($result.Length)"
} else {
    Write-Host "FAILED: No key found or empty blob"
}
